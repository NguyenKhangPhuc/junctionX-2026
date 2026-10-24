"""Ingestion pipeline: scan folders, parse, chunk, embed and persist knowledge vectors.

Deliberately incremental: files are fingerprinted with SHA-256 so unchanged files are skipped
on re-scan, and only changed/missing files are re-embedded or pruned.
"""

import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.config.settings import settings
from app.models.embedded_files import FileStatus
from app.models.embedded_folders import FolderKind, FolderStatus
from app.services.document_service import (
    DocumentParseError,
    DocumentService,
    DocumentTooLargeError,
    UnsupportedDocumentError,
    document_service,
)
from app.services.embedding_service import (
    EmbeddingDimensionError,
    EmbeddingUnavailableError,
    embedding_service,
)
from app.services.knowledge_store import KnowledgeStore, knowledge_store

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


class FolderNotFoundError(KeyError):
    """Raised when a folder registration id does not exist."""


class PathNotFoundError(FileNotFoundError):
    """Raised when a scan root or file path does not exist on disk."""


@dataclass
class FileResult:
    """Outcome of indexing a single file."""

    path: str
    status: str
    kind: str
    chunks: int = 0
    images: int = 0
    chars: int = 0
    error: str | None = None
    skipped_reason: str | None = None
    file_id: str | None = None
    embedding_id: str | None = None


@dataclass
class ScanStats:
    """Aggregate outcome of a folder (or whole corpus) indexing pass."""

    folder_id: str | None = None
    folder_path: str | None = None
    total_files: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    pruned: int = 0
    chunks: int = 0
    images: int = 0
    chars: int = 0
    elapsed_seconds: float = 0.0
    results: list[FileResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serializable summary without the per-file result list."""
        data = asdict(self)
        data.pop("results", None)
        return data


class IndexService:
    """Coordinates document parsing, chunking, embedding and knowledge-store writes."""

    def __init__(
        self,
        store: KnowledgeStore | None = None,
        documents: DocumentService | None = None,
        embeddings: Any | None = None,
    ) -> None:
        self.store = store or knowledge_store
        self.documents = documents or document_service
        self.embeddings = embeddings or embedding_service
        self._lock = threading.Lock()
        self._running = False

    # ------------------------------------------------------------------ helpers
    def _invalidate_lexical(self) -> None:
        """Tell the hybrid search service that its lexical corpus needs rebuilding."""
        try:
            from app.services.hybrid_search_service import hybrid_search_service

            hybrid_search_service.invalidate_lexical_index()
        except Exception as exc:  # noqa: BLE001 - invalidation is advisory
            logger.debug(f"Lexical index invalidation skipped: {exc}")

    @staticmethod
    def _is_junk(name: str) -> bool:
        """Skip hidden files, editor swap files and caches."""
        return name.startswith(".") or name.startswith("~") or name.lower() == "__pycache__"

    def scan_paths(self, root: str | Path, kind: FolderKind | str | None = None) -> list[Path]:
        """List indexable files under a directory.

        Args:
            root: Directory to walk.
            kind: Restrict to documents or images; None means both.

        Returns:
            list[Path]: Sorted supported files.

        Raises:
            PathNotFoundError: The root does not exist or is not a directory.
        """
        base = Path(root).expanduser()
        if not base.exists():
            raise PathNotFoundError(f"Path not found: {base}")
        if base.is_file():
            return [base.resolve()] if self.documents.is_supported(base) else []

        kind_value = FolderKind(kind) if isinstance(kind, str) else kind
        predicates: list[Callable[[Path], bool]] = []
        if kind_value in (None, FolderKind.DOCUMENT):
            predicates.append(self.documents.is_document)
        if kind_value in (None, FolderKind.IMAGE):
            predicates.append(self.documents.is_image)

        paths: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [name for name in dirnames if not self._is_junk(name)]
            for filename in filenames:
                if self._is_junk(filename):
                    continue
                candidate = Path(dirpath) / filename
                if candidate.is_file() and any(predicate(candidate) for predicate in predicates):
                    paths.append(candidate.resolve())
        return sorted(paths)

    def detect_kind(self, path: str | Path) -> str:
        """Folder/file kind for a path based on its extension."""
        return FolderKind.IMAGE.value if self.documents.is_image(path) else FolderKind.DOCUMENT.value

    # ------------------------------------------------------------------ single file
    def index_file(
        self,
        path: str | Path,
        *,
        kind: FolderKind | str | None = None,
        embedded_folder_id: str | None = None,
        force: bool = False,
        progress_cb: ProgressCallback | None = None,
        current: int = 0,
        total: int = 1,
    ) -> FileResult:
        """Index one file, reusing cached vectors when its hash is unchanged.

        Args:
            path: File path.
            kind: Explicit folder kind (defaults to extension-derived kind).
            embedded_folder_id: Owning folder registration id, when part of a folder scan.
            force: Re-index even when the content hash is unchanged.
            progress_cb: Optional ``(done, total, message)`` progress reporter.
            current: Progress counter passed to ``progress_cb``.
            total: Progress total passed to ``progress_cb``.

        Returns:
            FileResult: Outcome with status ready/skipped/failed.
        """
        file_path = Path(path).expanduser()
        if not file_path.is_file():
            return FileResult(
                path=str(file_path), status="failed", kind="document", error="File not found"
            )

        kind_value = FolderKind(kind) if isinstance(kind, str) else kind
        if kind_value is None:
            kind_value = FolderKind(self.detect_kind(file_path))
        mime_type = self.documents.mime_type_for(file_path)

        try:
            stat = file_path.stat()
            content_hash = self.documents.sha256(file_path)
        except OSError as exc:
            return FileResult(
                path=str(file_path),
                status="failed",
                kind=kind_value.value,
                error=f"Read error: {exc}",
            )

        resolved = str(file_path.resolve())
        existing = self.store.file_by_path(resolved)
        if (
            existing
            and not force
            and existing.get("status") == FileStatus.READY
            and existing.get("content_hash") == content_hash
        ):
            result = FileResult(
                path=resolved,
                status="skipped",
                kind=kind_value.value,
                chunks=int(existing.get("chunks_count") or 0),
                chars=int(existing.get("char_count") or 0),
                skipped_reason="unchanged",
                file_id=str(existing["id"]),
            )
            if progress_cb:
                progress_cb(current, total, f"skipped {file_path.name}")
            return result

        record = self.store.upsert_file(
            resolved,
            kind_value,
            embedded_folder_id=embedded_folder_id,
            mime_type=mime_type,
            size_bytes=stat.st_size,
            content_hash=content_hash,
            file_modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            status=FileStatus.RUNNING,
            attempt_count=int((existing or {}).get("attempt_count") or 0) + 1,
        )
        file_id = str(record["id"])

        try:
            if self.documents.is_image(file_path):
                result = self._index_image(file_path, file_id, kind_value, mime_type)
            else:
                result = self._index_document(file_path, file_id, kind_value)
        except (DocumentTooLargeError, UnsupportedDocumentError) as exc:
            result = FileResult(
                path=resolved,
                status="skipped",
                kind=kind_value.value,
                skipped_reason=str(exc),
                file_id=file_id,
            )
            self.store.update_file(
                file_id, status=FileStatus.SKIPPED, error_message=str(exc), chunks_count=0
            )
        except (
            DocumentParseError,
            EmbeddingUnavailableError,
            EmbeddingDimensionError,
            FileNotFoundError,
            OSError,
            ValueError,
        ) as exc:
            result = FileResult(
                path=resolved,
                status="failed",
                kind=kind_value.value,
                error=str(exc),
                file_id=file_id,
            )
            self.store.update_file(
                file_id, status=FileStatus.FAILED, error_message=str(exc), chunks_count=0
            )
            logger.warning(f"Indexing failed for {file_path.name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - unexpected parser/ONNX failures
            message = f"{exc.__class__.__name__}: {exc}"
            result = FileResult(
                path=resolved, status="failed", kind=kind_value.value, error=message, file_id=file_id
            )
            self.store.update_file(
                file_id, status=FileStatus.FAILED, error_message=message, chunks_count=0
            )
            logger.exception(f"Unexpected indexing failure for {file_path.name}")

        if progress_cb:
            progress_cb(current, total, f"{result.status} {file_path.name}")
        self._invalidate_lexical()
        return result

    def _index_document(
        self, file_path: Path, file_id: str, kind: FolderKind
    ) -> FileResult:
        """Parse, chunk and embed a text-bearing document."""
        parsed = self.documents.parse(file_path, max_bytes=settings.max_file_bytes)
        chunks = self.documents.chunk_document(
            parsed,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        if not chunks:
            self.store.update_file(
                file_id,
                status=FileStatus.SKIPPED,
                chunks_count=0,
                char_count=0,
                error_message="No extractable text",
            )
            return FileResult(
                path=parsed.path,
                status="skipped",
                kind=kind.value,
                skipped_reason="no extractable text",
                file_id=file_id,
            )

        vectors: list[list[float]] | None = None
        if self.embeddings.text_available():
            try:
                vectors = self.embeddings.embed_texts([chunk.text for chunk in chunks])
            except (EmbeddingUnavailableError, EmbeddingDimensionError) as exc:
                logger.warning(
                    f"Embedding failed for {file_path.name}; storing text only: {exc}"
                )
                vectors = None
        else:
            logger.warning(
                f"Text embedding model unavailable for {file_path.name}; storing text only"
            )

        stored = self.store.replace_file_chunks(
            file_id,
            chunks,
            vectors,
            model_name=self.embeddings.text_model,
            embed_dim=self.embeddings.text_dim,
        )
        return FileResult(
            path=parsed.path,
            status="ready",
            kind=kind.value,
            chunks=stored,
            chars=parsed.char_count,
            file_id=file_id,
        )

    def _index_image(
        self, file_path: Path, file_id: str, kind: FolderKind, mime_type: str
    ) -> FileResult:
        """Embed a raster image into the CLIP vector space."""
        if not self.embeddings.image_available():
            message = (
                f"Image embedding model '{self.embeddings.image_model}' unavailable; "
                "image stored without a vector"
            )
            self.store.update_file(
                file_id, status=FileStatus.SKIPPED, error_message=message, chunks_count=0
            )
            return FileResult(
                path=str(file_path),
                status="skipped",
                kind=kind.value,
                skipped_reason=message,
                file_id=file_id,
            )

        vector = self.embeddings.embed_image(str(file_path))
        image_id = self.store.replace_image(
            file_id,
            path=str(file_path),
            name=file_path.name,
            mime_type=mime_type,
            vector=vector,
            model_name=self.embeddings.image_model,
            embed_dim=self.embeddings.image_dim,
        )
        return FileResult(
            path=str(file_path),
            status="ready",
            kind=kind.value,
            images=1,
            file_id=file_id,
            embedding_id=str(image_id),
        )

    # ------------------------------------------------------------------ folder passes
    def register_folder(
        self,
        path: str | Path,
        kind: FolderKind | str = FolderKind.DOCUMENT,
        *,
        status: FolderStatus | str = FolderStatus.RUNNING,
    ) -> dict[str, Any]:
        """Register a folder and record how many indexable files it contains.

        Args:
            path: Folder path.
            kind: ``documents`` or ``images``.
            status: Initial folder status.

        Returns:
            dict[str, Any]: Folder record including ``total_files``.

        Raises:
            PathNotFoundError: The folder does not exist.
        """
        base = Path(path).expanduser()
        if not base.is_dir():
            raise PathNotFoundError(f"Folder not found: {base}")

        kind_value = FolderKind(kind) if isinstance(kind, str) else kind
        total = len(self.scan_paths(base, kind_value))
        return self.store.upsert_folder(
            str(base),
            kind_value,
            status=status,
            model_name=self.embeddings.text_model,
            embed_dim=self.embeddings.text_dim,
            total_files=total,
        )

    def index_folder(
        self,
        folder_id: str,
        *,
        force: bool = False,
        progress_cb: ProgressCallback | None = None,
    ) -> ScanStats:
        """Index or incrementally re-index a registered folder.

        Args:
            folder_id: Folder registration id.
            force: Re-embed every file regardless of content hash.
            progress_cb: Optional ``(done, total, message)`` progress reporter.

        Returns:
            ScanStats: Aggregate counters for the pass.

        Raises:
            FolderNotFoundError: No folder registration with that id.
        """
        folder = self.store.get_folder(folder_id)
        if folder is None:
            raise FolderNotFoundError(f"Unknown embedded folder: {folder_id}")

        folder_path = str(folder["folder_path"])
        kind_value = folder["kind"]
        started = time.monotonic()
        stats = ScanStats(folder_id=str(folder_id), folder_path=folder_path)

        with self._lock:
            if self._running:
                logger.info(f"Indexing pass already running for {folder_path}; waiting skipped")
            self._running = True
        try:
            paths = self.scan_paths(folder_path, kind_value)
            self.store.upsert_folder(
                folder_path,
                kind_value,
                status=FolderStatus.RUNNING,
                total_files=len(paths),
                model_name=self.embeddings.text_model,
                embed_dim=self.embeddings.text_dim,
            )
            stats.total_files = len(paths)

            present = {str(path) for path in paths}
            stats.pruned = self._prune_missing(folder_id, present)

            done = 0
            for index, path in enumerate(paths, start=1):
                result = self.index_file(
                    path,
                    kind=kind_value,
                    embedded_folder_id=str(folder_id),
                    force=force,
                    progress_cb=progress_cb,
                    current=index,
                    total=len(paths),
                )
                done = index
                stats.results.append(result)
                if result.status == "ready":
                    stats.processed += 1
                    stats.chunks += result.chunks
                    stats.images += result.images
                    stats.chars += result.chars
                elif result.status == "skipped":
                    stats.skipped += 1
                    stats.chunks += result.chunks
                    stats.chars += result.chars
                else:
                    stats.failed += 1
                    stats.errors.append(f"{Path(result.path).name}: {result.error}")

            counts = self.store.file_status_counts(folder_id)
            ready = counts.get(FileStatus.READY.value, 0)
            status = (
                FolderStatus.FAILED
                if stats.failed and ready == 0
                else FolderStatus.PARTIAL
                if stats.failed
                else FolderStatus.READY
            )
            self.store.upsert_folder(
                folder_path,
                kind_value,
                status=status,
                embedded_files=ready,
                total_files=len(paths),
                last_indexed_at=datetime.now(timezone.utc),
                model_name=self.embeddings.text_model,
                embed_dim=self.embeddings.text_dim,
            )
            stats.elapsed_seconds = round(time.monotonic() - started, 3)
        finally:
            with self._lock:
                self._running = False
            self._invalidate_lexical()

        return stats

    def _prune_missing(self, folder_id: str, present: set[str]) -> int:
        """Delete records for files that no longer exist on disk.

        Args:
            folder_id: Folder registration id.
            present: Paths seen in the current scan.

        Returns:
            int: Number of removed file records (and cascaded vectors).
        """
        removed = 0
        for record in self.store.list_files(embedded_folder_id=folder_id):
            path = str(record["path"])
            if path in present or Path(path).exists():
                continue
            self.store.delete_file(str(record["id"]))
            removed += 1
        return removed

    def index_all(
        self,
        *,
        force: bool = False,
        progress_cb: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Re-index every registered folder.

        Args:
            force: Re-embed all files even when unchanged.
            progress_cb: Optional progress reporter.

        Returns:
            dict[str, Any]: Combined summary across folders.
        """
        started = time.monotonic()
        totals = ScanStats()
        per_folder: list[dict[str, Any]] = []
        for folder in self.store.list_folders():
            stats = self.index_folder(str(folder["id"]), force=force, progress_cb=progress_cb)
            per_folder.append(stats.to_dict())
            totals.total_files += stats.total_files
            totals.processed += stats.processed
            totals.skipped += stats.skipped
            totals.failed += stats.failed
            totals.pruned += stats.pruned
            totals.chunks += stats.chunks
            totals.images += stats.images
            totals.chars += stats.chars
            totals.errors.extend(stats.errors)
        totals.elapsed_seconds = round(time.monotonic() - started, 3)
        payload = totals.to_dict()
        payload["folders"] = per_folder
        return payload

    def index_article(
        self,
        article_id: int,
        title: str,
        content: str,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> dict[str, Any]:
        """Embed a generated/curated article so later QA can cite it.

        Args:
            article_id: ``articles.id``.
            title: Article title.
            content: Article Markdown.
            chunk_size: Override the configured chunk size.
            chunk_overlap: Override the configured overlap.

        Returns:
            dict[str, Any]: ``{chunks, chars, vectors, model}`` summary.
        """
        chunks = self.documents.chunk_text(
            content,
            source_path=f"article://{article_id}",
            source_name=title,
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
            mime_type="text/markdown",
        )
        vectors: list[list[float]] | None = None
        if chunks and self.embeddings.text_available():
            try:
                vectors = self.embeddings.embed_texts([chunk.text for chunk in chunks])
            except (EmbeddingUnavailableError, EmbeddingDimensionError) as exc:
                logger.warning(f"Article embedding unavailable for article {article_id}: {exc}")
                vectors = None

        stored = self.store.replace_article_chunks(
            article_id,
            title,
            chunks,
            vectors,
            model_name=self.embeddings.text_model,
            embed_dim=self.embeddings.text_dim,
        )
        self._invalidate_lexical()
        return {
            "chunks": stored,
            "chars": sum(chunk.char_count for chunk in chunks),
            "vectors": stored if vectors is not None else 0,
            "model": self.embeddings.text_model,
        }

    def folder_summary(self, folder_id: str) -> dict[str, Any]:
        """Folder record with per-status file counts and chunk totals."""
        folder = self.store.get_folder(folder_id)
        if folder is None:
            raise FolderNotFoundError(f"Unknown embedded folder: {folder_id}")
        counts = self.store.file_status_counts(folder_id)
        return {
            "folder": folder,
            "file_status_counts": counts,
            "pending": sum(
                count
                for status, count in counts.items()
                if status in (FileStatus.QUEUED.value, FileStatus.RUNNING.value, FileStatus.STALE.value)
            ),
            "failed": counts.get(FileStatus.FAILED.value, 0),
            "ready": counts.get(FileStatus.READY.value, 0),
            "skipped": counts.get(FileStatus.SKIPPED.value, 0),
        }

    def corpus_stats(self) -> dict[str, Any]:
        """Corpus-wide counters used by the knowledge status endpoint."""
        counts = self.store.counts()
        embedding = self.embeddings.status()
        return {
            "backend": self.store.backend,
            "postgres_available": self.store.uses_postgres(),
            "folders": counts["folders"],
            "files": counts["files"],
            "ready_files": counts["ready_files"],
            "chunks": counts["chunks"],
            "chunks_with_vectors": counts["chunks_with_vectors"],
            "images": counts["images"],
            "embedded_chars": counts["embedded_chars"],
            "embedding": embedding,
            "settings": {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "max_file_bytes": settings.max_file_bytes,
                "text_embed_model": settings.text_embed_model,
                "text_embed_dim": settings.text_embed_dim,
                "image_embed_model": settings.image_embed_model,
                "image_embed_dim": settings.image_embed_dim,
            },
        }


index_service = IndexService()
