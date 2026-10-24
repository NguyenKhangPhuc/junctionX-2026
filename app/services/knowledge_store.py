"""Knowledge store abstraction over Postgres/pgvector with an in-process fallback backend.

Every retrieval and ingestion path goes through this store, so the intelligence core keeps
working (lexical search, folder registry, job bookkeeping) even when Postgres or pgvector is
unavailable. Writes are durable in Postgres; the memory backend keeps state per process.
"""

import logging
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from sqlalchemy import delete, func, insert, select
from sqlalchemy.orm import Session

from app.config.settings import settings
from app.models.embedded_files import EmbeddedFiles, FileStatus
from app.models.embedded_folders import EmbeddedFolders, FolderKind, FolderStatus
from app.models.embeddings import (
    DocumentChunk,
    ImageEmbeddingRecord,
    SourceType,
)
from app.utils.database import database_is_available, get_db_session, safe_schema_bootstrap

logger = logging.getLogger(__name__)


def _now() -> datetime:
    """Timezone-aware timestamp used for memory-backend bookkeeping."""
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    """ISO-8601 string for a datetime, or None."""
    return value.isoformat() if value else None


def _as_id(value: Any) -> uuid.UUID:
    """Coerce str/uuid values into a uuid.UUID."""
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors (0.0 when either has zero norm)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class StoredChunk:
    """A retrievable text chunk with provenance and its dense vector (if loaded)."""

    id: uuid.UUID
    source_type: SourceType
    source_path: str
    source_name: str
    mime_type: str
    chunk_index: int
    text: str
    char_count: int
    page_number: int | None = None
    section_label: str | None = None
    model_name: str = ""
    embed_dim: int = 0
    vector: list[float] | None = None
    has_vector: bool = True
    embedded_file_id: uuid.UUID | None = None
    article_id: int | None = None

    def citation(self, marker: str) -> dict[str, Any]:
        """Build the citation payload persisted alongside generated articles.

        Args:
            marker: Evidence marker such as ``E1``.

        Returns:
            dict[str, Any]: Citation record with source path, page and section labels.
        """
        return {
            "marker": marker,
            "chunk_id": str(self.id),
            "source_type": self.source_type.value,
            "path": self.source_path,
            "name": self.source_name,
            "page": self.page_number,
            "section": self.section_label,
            "chunk_index": self.chunk_index,
        }


@dataclass
class StoredImage:
    """A CLIP-indexed image with its vector and provenance."""

    id: uuid.UUID
    path: str
    name: str
    mime_type: str
    model_name: str = ""
    embed_dim: int = 0
    vector: list[float] | None = None
    caption: str | None = None
    embedded_file_id: uuid.UUID | None = None

    def citation(self, marker: str) -> dict[str, Any]:
        """Build a citation record for an image hit."""
        return {
            "marker": marker,
            "image_id": str(self.id),
            "source_type": "image",
            "path": self.path,
            "name": self.name,
            "caption": self.caption,
        }


@dataclass
class _MemoryState:
    """Process-local fallback state mirroring the Postgres tables."""

    folders: dict[uuid.UUID, dict[str, Any]] = field(default_factory=dict)
    folder_by_path: dict[str, uuid.UUID] = field(default_factory=dict)
    files: dict[uuid.UUID, dict[str, Any]] = field(default_factory=dict)
    file_by_path: dict[str, uuid.UUID] = field(default_factory=dict)
    chunks: dict[uuid.UUID, StoredChunk] = field(default_factory=dict)
    images: dict[uuid.UUID, StoredImage] = field(default_factory=dict)


class KnowledgeStore:
    """Unified read/write surface for folders, files, chunks and image embeddings."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._memory = _MemoryState()

    # ------------------------------------------------------------------ backend selection
    @property
    def backend(self) -> str:
        """Active backend name: ``postgres`` or ``memory``."""
        return "postgres" if database_is_available() else "memory"

    def uses_postgres(self) -> bool:
        """True when persistent vector storage is reachable."""
        return self.backend == "postgres"

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Yield a database session; callers must guard with :meth:`uses_postgres` first."""
        with get_db_session() as session:
            yield session

    def ensure_schema(self) -> bool:
        """Create missing tables/extensions when possible (idempotent)."""
        return safe_schema_bootstrap()

    def clear_memory(self) -> None:
        """Drop all process-local fallback state (used by tests)."""
        with self._lock:
            self._memory = _MemoryState()

    # ------------------------------------------------------------------ folders
    def upsert_folder(
        self,
        path: str,
        kind: FolderKind | str,
        *,
        model_name: str | None = None,
        embed_dim: int | None = None,
        status: FolderStatus | str | None = None,
        total_files: int | None = None,
        embedded_files: int | None = None,
        failed_files: int | None = None,
        skipped_files: int | None = None,
        last_scanned_at: datetime | None = None,
        last_indexed_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Insert or update the folder registration record for a directory path."""
        resolved = str(Path(path).expanduser().resolve())
        kind_value = FolderKind(kind) if not isinstance(kind, FolderKind) else kind
        status_value = (
            FolderStatus(status)
            if status is not None and not isinstance(status, FolderStatus)
            else status
        )

        if self.uses_postgres():
            with self.session() as session:
                folder = session.execute(
                    select(EmbeddedFolders).where(
                        EmbeddedFolders.path == resolved,
                        EmbeddedFolders.kind == kind_value,
                    )
                ).scalar_one_or_none()
                if folder is None:
                    folder = EmbeddedFolders(
                        path=resolved,
                        kind=kind_value,
                        status=status_value or FolderStatus.RUNNING,
                        model_name=model_name or settings.text_embed_model,
                        embed_dim=embed_dim if embed_dim is not None else settings.text_embed_dim,
                        files_total=total_files or 0,
                        files_embedded=embedded_files or 0,
                        files_failed=failed_files or 0,
                        files_skipped=skipped_files or 0,
                        last_scanned_at=last_scanned_at,
                        last_embedded_at=last_indexed_at,
                    )
                    session.add(folder)
                else:
                    if model_name is not None:
                        folder.model_name = model_name
                    if embed_dim is not None:
                        folder.embed_dim = embed_dim
                    if status_value is not None:
                        folder.status = status_value
                    if total_files is not None:
                        folder.files_total = total_files
                    if embedded_files is not None:
                        folder.files_embedded = embedded_files
                    if failed_files is not None:
                        folder.files_failed = failed_files
                    if skipped_files is not None:
                        folder.files_skipped = skipped_files
                    if last_scanned_at is not None:
                        folder.last_scanned_at = last_scanned_at
                    if last_indexed_at is not None:
                        folder.last_embedded_at = last_indexed_at
                session.flush()
                return self._folder_dict(folder)

        with self._lock:
            existing_id = self._memory.folder_by_path.get(resolved)
            record = self._memory.folders.get(existing_id, {}) if existing_id else {}
            folder_id = existing_id or uuid.uuid4()
            record.update(
                {
                    "id": folder_id,
                    "path": resolved,
                    "folder_path": resolved,
                    "kind": kind_value,
                    "status": status_value or FolderStatus.RUNNING,
                    "model_name": model_name if model_name is not None else record.get("model_name"),
                    "embed_dim": embed_dim if embed_dim is not None else record.get("embed_dim"),
                    "total_files": total_files
                    if total_files is not None
                    else record.get("total_files", 0),
                    "embedded_files": embedded_files
                    if embedded_files is not None
                    else record.get("embedded_files", 0),
                    "failed_files": failed_files
                    if failed_files is not None
                    else record.get("failed_files", 0),
                    "skipped_files": skipped_files
                    if skipped_files is not None
                    else record.get("skipped_files", 0),
                    "last_scanned_at": last_scanned_at
                    if last_scanned_at is not None
                    else record.get("last_scanned_at"),
                    "last_indexed_at": last_indexed_at
                    if last_indexed_at is not None
                    else record.get("last_indexed_at"),
                    "created_at": record.get("created_at") or _now(),
                    "updated_at": _now(),
                }
            )
            self._memory.folders[folder_id] = record
            self._memory.folder_by_path[resolved] = folder_id
            return dict(record)

    def get_folder(self, folder_id: str | uuid.UUID) -> dict[str, Any] | None:
        """Fetch a folder registration by id."""
        folder_uuid = _as_id(folder_id)
        if self.uses_postgres():
            with self.session() as session:
                folder = session.get(EmbeddedFolders, folder_uuid)
                return self._folder_dict(folder) if folder else None
        with self._lock:
            record = self._memory.folders.get(folder_uuid)
            return dict(record) if record else None

    def folder_by_path(self, path: str) -> dict[str, Any] | None:
        """Fetch a folder registration by directory path."""
        resolved = str(Path(path).expanduser().resolve())
        if self.uses_postgres():
            with self.session() as session:
                folder = session.execute(
                    select(EmbeddedFolders).where(EmbeddedFolders.path == resolved)
                ).scalar_one_or_none()
                return self._folder_dict(folder) if folder else None
        with self._lock:
            folder_id = self._memory.folder_by_path.get(resolved)
            record = self._memory.folders.get(folder_id) if folder_id else None
            return dict(record) if record else None

    def list_folders(self, kind: FolderKind | str | None = None) -> list[dict[str, Any]]:
        """List registered folders, newest first."""
        kind_value = FolderKind(kind) if isinstance(kind, str) else kind
        if self.uses_postgres():
            with self.session() as session:
                statement = select(EmbeddedFolders).order_by(EmbeddedFolders.created_at.desc())
                if kind_value is not None:
                    statement = statement.where(EmbeddedFolders.kind == kind_value)
                return [self._folder_dict(folder) for folder in session.execute(statement)]
        with self._lock:
            records = sorted(
                self._memory.folders.values(),
                key=lambda item: item.get("created_at") or _now(),
                reverse=True,
            )
            return [
                dict(record)
                for record in records
                if kind_value is None or record.get("kind") == kind_value
            ]

    def delete_folder(self, folder_id: str | uuid.UUID) -> bool:
        """Delete a folder registration and cascade its file records."""
        folder_uuid = _as_id(folder_id)
        if self.uses_postgres():
            with self.session() as session:
                folder = session.get(EmbeddedFolders, folder_uuid)
                if folder is None:
                    return False
                session.delete(folder)
                return True
        with self._lock:
            record = self._memory.folders.pop(folder_uuid, None)
            if record is None:
                return False
            self._memory.folder_by_path.pop(record.get("folder_path") or record.get("path", ""), None)
            for file_id, file_record in list(self._memory.files.items()):
                if file_record.get("embedded_folder_id") == folder_uuid:
                    self.forget_file(file_id)
            return True

    # ------------------------------------------------------------------ files
    def upsert_file(
        self,
        path: str,
        kind: FolderKind | str,
        *,
        embedded_folder_id: str | uuid.UUID | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        content_hash: str | None = None,
        file_modified_at: datetime | None = None,
        status: FileStatus | str = FileStatus.QUEUED,
        model_name: str | None = None,
        error_message: str | None = None,
        attempt_count: int | None = None,
    ) -> dict[str, Any]:
        """Insert or update the file index record for a single path."""
        resolved = str(Path(path).expanduser().resolve())
        kind_value = FolderKind(kind) if not isinstance(kind, FolderKind) else kind
        status_value = FileStatus(status) if not isinstance(status, FileStatus) else status
        file_uuid = self._existing_file_id(resolved)
        folder_uuid = _as_id(embedded_folder_id) if embedded_folder_id is not None else None

        if self.uses_postgres():
            with self.session() as session:
                row = session.get(EmbeddedFiles, file_uuid) if file_uuid else None
                if row is None:
                    row = EmbeddedFiles(
                        id=file_uuid or uuid.uuid4(),
                        embedded_folder_id=folder_uuid,
                        path=resolved,
                        name=Path(resolved).name,
                        extension=Path(resolved).suffix.lower(),
                        mime_type=mime_type or "application/octet-stream",
                        size_bytes=size_bytes or 0,
                        content_hash=content_hash,
                        file_modified_at=file_modified_at,
                        status=status_value,
                        model_name=model_name,
                        error_message=error_message,
                        attempt_count=attempt_count or 0,
                    )
                    session.add(row)
                else:
                    row.kind = kind_value
                    if folder_uuid is not None:
                        row.embedded_folder_id = folder_uuid
                    if mime_type is not None:
                        row.mime_type = mime_type
                    if size_bytes is not None:
                        row.size_bytes = size_bytes
                    if content_hash is not None:
                        row.content_hash = content_hash
                    if file_modified_at is not None:
                        row.file_modified_at = file_modified_at
                    row.status = status_value
                    if model_name is not None:
                        row.model_name = model_name
                    if error_message is not None:
                        row.error_message = error_message
                    if attempt_count is not None:
                        row.attempt_count = attempt_count
                session.flush()
                return self._file_dict(row)

        with self._lock:
            record = self._memory.files.get(file_uuid) if file_uuid else {}
            record_id = file_uuid or uuid.uuid4()
            record.update(
                {
                    "id": record_id,
                    "embedded_folder_id": folder_uuid or record.get("embedded_folder_id"),
                    "path": resolved,
                    "name": Path(resolved).name,
                    "extension": Path(resolved).suffix.lower(),
                    "kind": kind_value,
                    "mime_type": mime_type or record.get("mime_type") or "application/octet-stream",
                    "size_bytes": size_bytes if size_bytes is not None else record.get("size_bytes", 0),
                    "content_hash": content_hash if content_hash is not None else record.get("content_hash"),
                    "file_modified_at": file_modified_at or record.get("file_modified_at"),
                    "status": status_value,
                    "error_message": error_message if error_message is not None else record.get("error_message"),
                    "attempt_count": attempt_count
                    if attempt_count is not None
                    else record.get("attempt_count", 0),
                    "model_name": model_name if model_name is not None else record.get("model_name"),
                    "char_count": record.get("char_count", 0),
                    "chunks_count": record.get("chunks_count", 0),
                    "created_at": record.get("created_at") or _now(),
                    "updated_at": _now(),
                }
            )
            self._memory.files[record_id] = record
            self._memory.file_by_path[resolved] = record_id
            return dict(record)

    def _existing_file_id(self, resolved_path: str) -> uuid.UUID | None:
        """Look up the file record id for a path, in any backend."""
        if self.uses_postgres():
            with self.session() as session:
                return session.execute(
                    select(EmbeddedFiles.id).where(EmbeddedFiles.path == resolved_path)
                ).scalar_one_or_none()
        with self._lock:
            return self._memory.file_by_path.get(resolved_path)

    def get_file(self, file_id: str | uuid.UUID) -> dict[str, Any] | None:
        """Fetch a file record by id."""
        file_uuid = _as_id(file_id)
        if self.uses_postgres():
            with self.session() as session:
                row = session.get(EmbeddedFiles, file_uuid)
                return self._file_dict(row) if row else None
        with self._lock:
            record = self._memory.files.get(file_uuid)
            return dict(record) if record else None

    def file_by_path(self, path: str) -> dict[str, Any] | None:
        """Fetch a file record by filesystem path."""
        resolved = str(Path(path).expanduser().resolve())
        if self.uses_postgres():
            with self.session() as session:
                row = session.execute(
                    select(EmbeddedFiles).where(EmbeddedFiles.path == resolved)
                ).scalar_one_or_none()
                return self._file_dict(row) if row else None
        with self._lock:
            file_id = self._memory.file_by_path.get(resolved)
            record = self._memory.files.get(file_id) if file_id else None
            return dict(record) if record else None

    def update_file(self, file_id: str | uuid.UUID, **fields: Any) -> dict[str, Any] | None:
        """Apply field updates to a file record (``status``, counters, errors, timestamps)."""
        file_uuid = _as_id(file_id)
        if self.uses_postgres():
            with self.session() as session:
                row = session.get(EmbeddedFiles, file_uuid)
                if row is None:
                    return None
                for key, value in fields.items():
                    setattr(row, key, value)
                session.flush()
                return self._file_dict(row)
        with self._lock:
            record = self._memory.files.get(file_uuid)
            if record is None:
                return None
            record.update(fields)
            record["updated_at"] = _now()
            return dict(record)

    def list_files(
        self,
        *,
        embedded_folder_id: str | uuid.UUID | None = None,
        statuses: Sequence[FileStatus | str] | None = None,
        kind: FolderKind | str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """List file records with optional folder/status/kind filters."""
        status_values = [FileStatus(s) if isinstance(s, str) else s for s in statuses] if statuses else None
        kind_value = FolderKind(kind) if isinstance(kind, str) else kind
        folder_uuid = _as_id(embedded_folder_id) if embedded_folder_id is not None else None

        if self.uses_postgres():
            with self.session() as session:
                statement = select(EmbeddedFiles).order_by(EmbeddedFiles.created_at.desc())
                if folder_uuid is not None:
                    statement = statement.where(EmbeddedFiles.embedded_folder_id == folder_uuid)
                if status_values:
                    statement = statement.where(EmbeddedFiles.status.in_(status_values))
                if kind_value is not None:
                    statement = statement.where(EmbeddedFiles.kind == kind_value)
                if limit is not None:
                    statement = statement.limit(limit)
                return [self._file_dict(row) for row in session.execute(statement)]

        with self._lock:
            records = sorted(
                self._memory.files.values(),
                key=lambda item: item.get("created_at") or _now(),
                reverse=True,
            )
            filtered = [
                record
                for record in records
                if (folder_uuid is None or record.get("embedded_folder_id") == folder_uuid)
                and (not status_values or record.get("status") in status_values)
                and (kind_value is None or record.get("kind") == kind_value)
            ]
            return [dict(record) for record in filtered[: limit if limit else len(filtered)]]

    def forget_file(self, file_id: uuid.UUID) -> None:
        """Remove a file record from the memory backend (cascading its vectors)."""
        with self._lock:
            record = self._memory.files.pop(file_id, None)
            if record is None:
                return
            self._memory.file_by_path.pop(record.get("path", ""), None)
            for chunk_id, chunk in list(self._memory.chunks.items()):
                if chunk.embedded_file_id == file_id:
                    self._memory.chunks.pop(chunk_id, None)
            for image_id, image in list(self._memory.images.items()):
                if image.embedded_file_id == file_id:
                    self._memory.images.pop(image_id, None)

    def delete_file(self, file_id: str | uuid.UUID) -> bool:
        """Delete a file record together with its chunk/image vectors."""
        file_uuid = _as_id(file_id)
        if self.uses_postgres():
            with self.session() as session:
                row = session.get(EmbeddedFiles, file_uuid)
                if row is None:
                    return False
                session.delete(row)
                return True
        with self._lock:
            existed = file_uuid in self._memory.files
            self.forget_file(file_uuid)
            return existed

    # ------------------------------------------------------------------ chunks
    def replace_file_chunks(
        self,
        file_id: str | uuid.UUID,
        chunks: Sequence[Any],
        vectors: Sequence[Sequence[float]] | None,
        *,
        model_name: str,
        embed_dim: int,
    ) -> int:
        """Replace the chunk set of a file with freshly embedded chunks.

        Args:
            file_id: Owning ``embedded_files`` id.
            chunks: :class:`~app.services.document_service.ChunkRecord` objects.
            vectors: Dense vectors aligned with ``chunks``, or None when the embedding model
                was unavailable (text is still stored for lexical retrieval).
            model_name: Embedding model name to record.
            embed_dim: Vector width to record.

        Returns:
            int: Number of stored chunks.
        """
        file_uuid = _as_id(file_id)
        file_record = self.get_file(file_uuid)
        if file_record is None:
            raise KeyError(f"Unknown embedded file id: {file_id}")
        source_path = file_record["path"]
        source_name = file_record["name"]
        mime_type = file_record.get("mime_type") or "text/plain"

        if vectors is not None and len(chunks) != len(vectors):
            raise ValueError(
                f"Chunk/vector length mismatch: {len(chunks)} chunks, {len(vectors)} vectors"
            )
        zero_vector = [0.0] * embed_dim
        aligned: list[Sequence[float] | None] = list(vectors) if vectors is not None else [None] * len(chunks)

        if self.uses_postgres():
            with self.session() as session:
                session.execute(
                    delete(DocumentChunk).where(
                        DocumentChunk.embedded_file_id == file_uuid,
                        DocumentChunk.source_type == SourceType.FILE,
                    )
                )
                values = [
                    {
                        "id": uuid.uuid4(),
                        "embedded_file_id": file_uuid,
                        "source_type": SourceType.FILE,
                        "source_path": source_path,
                        "source_name": source_name,
                        "mime_type": mime_type,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "char_count": chunk.char_count,
                        "page_number": chunk.page_number,
                        "section_label": chunk.section_label,
                        "model_name": model_name,
                        "embed_dim": embed_dim,
                        "vector": list(vector) if vector is not None else zero_vector,
                        "has_vector": vector is not None,
                    }
                    for chunk, vector in zip(chunks, aligned)
                ]
                if values:
                    session.execute(insert(DocumentChunk), values)
                self._bump_file_indexed(
                    session, file_uuid, chunk_total=len(values), is_image=False
                )
                return len(values)

        with self._lock:
            for chunk_id, chunk in list(self._memory.chunks.items()):
                if chunk.embedded_file_id == file_uuid and chunk.source_type == SourceType.FILE:
                    self._memory.chunks.pop(chunk_id, None)
            for chunk, vector in zip(chunks, aligned):
                stored = StoredChunk(
                    id=uuid.uuid4(),
                    source_type=SourceType.FILE,
                    source_path=source_path,
                    source_name=source_name,
                    mime_type=mime_type,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    char_count=chunk.char_count,
                    page_number=chunk.page_number,
                    section_label=chunk.section_label,
                    model_name=model_name,
                    embed_dim=embed_dim,
                    vector=list(vector) if vector is not None else None,
                    has_vector=vector is not None,
                    embedded_file_id=file_uuid,
                )
                self._memory.chunks[stored.id] = stored
            record = self._memory.files[file_uuid]
            record["chunks_count"] = len(chunks)
            record["char_count"] = sum(chunk.char_count for chunk in chunks)
            record["model_name"] = model_name
            record["updated_at"] = _now()
            return len(chunks)

    def replace_article_chunks(
        self,
        article_id: int,
        title: str,
        chunks: Sequence[Any],
        vectors: Sequence[Sequence[float]] | None,
        *,
        model_name: str,
        embed_dim: int,
    ) -> int:
        """Replace the chunk set of a generated/curated article."""
        if vectors is not None and len(chunks) != len(vectors):
            raise ValueError(
                f"Chunk/vector length mismatch: {len(chunks)} chunks, {len(vectors)} vectors"
            )
        zero_vector = [0.0] * embed_dim
        aligned: list[Sequence[float] | None] = (
            list(vectors) if vectors is not None else [None] * len(chunks)
        )

        if self.uses_postgres():
            with self.session() as session:
                session.execute(
                    delete(DocumentChunk).where(
                        DocumentChunk.article_id == article_id,
                        DocumentChunk.source_type == SourceType.ARTICLE,
                    )
                )
                values = [
                    {
                        "id": uuid.uuid4(),
                        "article_id": article_id,
                        "source_type": SourceType.ARTICLE,
                        "source_path": f"article://{article_id}",
                        "source_name": title,
                        "mime_type": "text/markdown",
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "char_count": chunk.char_count,
                        "page_number": chunk.page_number,
                        "section_label": chunk.section_label,
                        "model_name": model_name,
                        "embed_dim": embed_dim,
                        "vector": list(vector) if vector is not None else zero_vector,
                        "has_vector": vector is not None,
                    }
                    for chunk, vector in zip(chunks, aligned)
                ]
                if values:
                    session.execute(insert(DocumentChunk), values)
                return len(values)

        with self._lock:
            for chunk_id, chunk in list(self._memory.chunks.items()):
                if chunk.article_id == article_id and chunk.source_type == SourceType.ARTICLE:
                    self._memory.chunks.pop(chunk_id, None)
            for chunk, vector in zip(chunks, aligned):
                stored = StoredChunk(
                    id=uuid.uuid4(),
                    source_type=SourceType.ARTICLE,
                    source_path=f"article://{article_id}",
                    source_name=title,
                    mime_type="text/markdown",
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    char_count=chunk.char_count,
                    page_number=chunk.page_number,
                    section_label=chunk.section_label,
                    model_name=model_name,
                    embed_dim=embed_dim,
                    vector=list(vector) if vector is not None else None,
                    has_vector=vector is not None,
                    article_id=article_id,
                )
                self._memory.chunks[stored.id] = stored
            return len(chunks)

    def iter_chunks(
        self,
        *,
        source_types: Sequence[SourceType | str] | None = None,
        with_vectors: bool = True,
    ) -> list[StoredChunk]:
        """Load chunks for lexical indexing or vector search.

        Args:
            source_types: Restrict to file/article sources.
            with_vectors: When false, vectors are omitted to save memory and I/O.

        Returns:
            list[StoredChunk]: Loaded chunk records.
        """
        types = [SourceType(s) if isinstance(s, str) else s for s in source_types] if source_types else None

        if self.uses_postgres():
            with self.session() as session:
                statement = select(DocumentChunk)
                if types:
                    statement = statement.where(DocumentChunk.source_type.in_(types))
                rows = list(session.execute(statement))
                return [self._chunk_dict_to_stored(row, with_vectors=with_vectors) for row in rows]

        with self._lock:
            chunks = [
                chunk
                for chunk in self._memory.chunks.values()
                if not types or chunk.source_type in types
            ]
            if with_vectors:
                return list(chunks)
            return [replace(chunk, vector=None) for chunk in chunks]

    def get_chunk(self, chunk_id: str | uuid.UUID) -> StoredChunk | None:
        """Fetch a single chunk (with its vector) for evidence resolution."""
        chunk_uuid = _as_id(chunk_id)
        if self.uses_postgres():
            with self.session() as session:
                row = session.get(DocumentChunk, chunk_uuid)
                return self._chunk_dict_to_stored(row) if row else None
        with self._lock:
            chunk = self._memory.chunks.get(chunk_uuid)
            return chunk if chunk else None

    def count_chunks(self) -> int:
        """Total stored chunks."""
        if self.uses_postgres():
            with self.session() as session:
                return int(session.execute(select(func.count(DocumentChunk.id))).scalar() or 0)
        with self._lock:
            return len(self._memory.chunks)

    def vector_search_chunks(
        self,
        query_vector: Sequence[float],
        k: int = 12,
        *,
        source_types: Sequence[SourceType | str] | None = None,
        max_distance: float | None = None,
    ) -> list[tuple[StoredChunk, float]]:
        """Nearest-neighbour chunk search using pgvector cosine distance or numpy fallback.

        Args:
            query_vector: Dense query embedding.
            k: Maximum hits.
            source_types: Optional file/article filter.
            max_distance: Optional cosine distance cutoff (score cutoff = 1 - max_distance).

        Returns:
            list[tuple[StoredChunk, float]]: Hits with similarity scores in descending order.
        """
        if k <= 0:
            return []
        types = [SourceType(s) if isinstance(s, str) else s for s in source_types] if source_types else None

        if self.uses_postgres():
            with self.session() as session:
                distance = DocumentChunk.vector.cosine_distance(query_vector).label("distance")
                statement = (
                    select(DocumentChunk, distance)
                    .where(DocumentChunk.has_vector.is_(True))
                    .order_by(distance)
                )
                if types:
                    statement = statement.where(DocumentChunk.source_type.in_(types))
                if max_distance is not None:
                    statement = statement.where(distance < max_distance)
                statement = statement.limit(k)
                results: list[tuple[StoredChunk, float]] = []
                for row, cosine_distance in session.execute(statement):
                    results.append(
                        (self._chunk_dict_to_stored(row), 1.0 - float(cosine_distance))
                    )
                return results

        scored: list[tuple[StoredChunk, float]] = []
        with self._lock:
            for chunk in self._memory.chunks.values():
                if types and chunk.source_type not in types:
                    continue
                if chunk.vector is None or not chunk.has_vector:
                    continue
                similarity = _cosine_similarity(query_vector, chunk.vector)
                if max_distance is not None and similarity < 1.0 - max_distance:
                    continue
                scored.append((chunk, similarity))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    # ------------------------------------------------------------------ images
    def replace_image(
        self,
        file_id: str | uuid.UUID,
        path: str,
        name: str,
        mime_type: str,
        vector: Sequence[float],
        *,
        model_name: str,
        embed_dim: int,
        caption: str | None = None,
    ) -> uuid.UUID:
        """Store (or replace) the CLIP vector for an image file."""
        file_uuid = _as_id(file_id)
        resolved = str(Path(path).expanduser().resolve())

        if self.uses_postgres():
            with self.session() as session:
                session.execute(
                    delete(ImageEmbeddingRecord).where(
                        ImageEmbeddingRecord.embedded_file_id == file_uuid
                    )
                )
                image_id = uuid.uuid4()
                session.add(
                    ImageEmbeddingRecord(
                        id=image_id,
                        embedded_file_id=file_uuid,
                        path=resolved,
                        name=name,
                        mime_type=mime_type,
                        caption=caption,
                        model_name=model_name,
                        embed_dim=embed_dim,
                        vector=list(vector),
                    )
                )
                self._bump_file_indexed(session, file_uuid, chunk_total=1, is_image=True)
                return image_id

        with self._lock:
            for image_id, image in list(self._memory.images.items()):
                if image.embedded_file_id == file_uuid:
                    self._memory.images.pop(image_id, None)
            image_id = uuid.uuid4()
            self._memory.images[image_id] = StoredImage(
                id=image_id,
                path=resolved,
                name=name,
                mime_type=mime_type,
                model_name=model_name,
                embed_dim=embed_dim,
                vector=list(vector),
                caption=caption,
                embedded_file_id=file_uuid,
            )
            record = self._memory.files.get(file_uuid)
            if record is not None:
                record["chunks_count"] = max(1, record.get("chunks_count", 0))
                record["model_name"] = model_name
                record["updated_at"] = _now()
            return image_id

    def iter_images(self, with_vectors: bool = True) -> list[StoredImage]:
        """Load image records for lexical/vector use."""
        if self.uses_postgres():
            with self.session() as session:
                rows = list(session.execute(select(ImageEmbeddingRecord)))
                images = [self._image_to_stored(row, with_vectors=with_vectors) for row in rows]
                return images
        with self._lock:
            if with_vectors:
                return list(self._memory.images.values())
            return [replace(image, vector=None) for image in self._memory.images.values()]

    def get_image(self, image_id: str | uuid.UUID) -> StoredImage | None:
        """Fetch one image record by id."""
        image_uuid = _as_id(image_id)
        if self.uses_postgres():
            with self.session() as session:
                row = session.get(ImageEmbeddingRecord, image_uuid)
                return self._image_to_stored(row) if row else None
        with self._lock:
            return self._memory.images.get(image_uuid)

    def count_images(self) -> int:
        """Total stored image embeddings."""
        if self.uses_postgres():
            with self.session() as session:
                return int(
                    session.execute(select(func.count(ImageEmbeddingRecord.id))).scalar() or 0
                )
        with self._lock:
            return len(self._memory.images)

    def vector_search_images(
        self,
        query_vector: Sequence[float],
        k: int = 8,
        *,
        max_distance: float | None = None,
    ) -> list[tuple[StoredImage, float]]:
        """Nearest-neighbour image search with CLIP text-query vectors."""
        if k <= 0:
            return []

        if self.uses_postgres():
            with self.session() as session:
                distance = ImageEmbeddingRecord.vector.cosine_distance(query_vector).label(
                    "distance"
                )
                statement = select(ImageEmbeddingRecord, distance).order_by(distance)
                if max_distance is not None:
                    statement = statement.where(distance < max_distance)
                statement = statement.limit(k)
                return [
                    (self._image_to_stored(row), 1.0 - float(cosine_distance))
                    for row, cosine_distance in session.execute(statement)
                ]

        scored: list[tuple[StoredImage, float]] = []
        with self._lock:
            for image in self._memory.images.values():
                if image.vector is None:
                    continue
                similarity = _cosine_similarity(query_vector, image.vector)
                if max_distance is not None and similarity < 1.0 - max_distance:
                    continue
                scored.append((image, similarity))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    # ------------------------------------------------------------------ aggregate stats
    def counts(self) -> dict[str, Any]:
        """Counts of indexed artifacts plus the active backend name."""
        if self.uses_postgres():
            with self.session() as session:
                files_total = int(session.execute(select(func.count(EmbeddedFiles.id))).scalar() or 0)
                ready_files = int(
                    session.execute(
                        select(func.count(EmbeddedFiles.id)).where(
                            EmbeddedFiles.status == FileStatus.READY
                        )
                    ).scalar()
                    or 0
                )
                folders_total = int(
                    session.execute(select(func.count(EmbeddedFolders.id))).scalar() or 0
                )
                return {
                    "backend": "postgres",
                    "folders": folders_total,
                    "files": files_total,
                    "ready_files": ready_files,
                    "chunks": int(session.execute(select(func.count(DocumentChunk.id))).scalar() or 0),
                    "images": int(
                        session.execute(select(func.count(ImageEmbeddingRecord.id))).scalar() or 0
                    ),
                }

        with self._lock:
            ready = sum(
                1 for record in self._memory.files.values() if record.get("status") == FileStatus.READY
            )
            return {
                "backend": "memory",
                "folders": len(self._memory.folders),
                "files": len(self._memory.files),
                "ready_files": ready,
                "chunks": len(self._memory.chunks),
                "images": len(self._memory.images),
            }

    def file_status_counts(self, folder_id: str | uuid.UUID | None = None) -> dict[str, int]:
        """Per-status file counters for folder/embed status payloads.

        Args:
            folder_id: Restrict counters to one folder; None counts the whole corpus.

        Returns:
            dict[str, int]: Status name to count, plus a ``total`` key.
        """
        folder_uuid = _as_id(folder_id) if folder_id is not None else None
        if self.uses_postgres():
            with self.session() as session:
                statement = select(EmbeddedFiles.status, func.count(EmbeddedFiles.id)).group_by(
                    EmbeddedFiles.status
                )
                if folder_uuid is not None:
                    statement = statement.where(EmbeddedFiles.embedded_folder_id == folder_uuid)
                rows = session.execute(statement)
                counters = {
                    str(status.value if hasattr(status, "value") else status): int(total)
                    for status, total in rows
                }
                counters["total"] = sum(
                    value for key, value in counters.items() if key != "total"
                )
                return counters
        with self._lock:
            counters = {}
            for record in self._memory.files.values():
                if folder_uuid is not None and record.get("embedded_folder_id") != folder_uuid:
                    continue
                status = record.get("status")
                key = status.value if isinstance(status, FileStatus) else str(status)
                counters[key] = counters.get(key, 0) + 1
            counters["total"] = sum(value for key, value in counters.items() if key != "total")
            return counters

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _bump_file_indexed(
        session: Session, file_uuid: uuid.UUID, chunk_total: int, *, is_image: bool
    ) -> None:
        """Mark a file as READY and record its vector/chunk counters."""
        row = session.get(EmbeddedFiles, file_uuid)
        if row is None:
            return
        row.status = FileStatus.READY
        row.chunks_count = chunk_total
        row.error_message = None
        if not is_image:
            aggregate = session.execute(
                select(func.coalesce(func.sum(DocumentChunk.char_count), 0)).where(
                    DocumentChunk.embedded_file_id == file_uuid
                )
            ).scalar_one()
            row.char_count = int(aggregate or 0)

    @staticmethod
    def _folder_dict(folder: EmbeddedFolders) -> dict[str, Any]:
        """Convert a folder ORM row into a plain dict (both ``path`` and ``folder_path`` keys)."""
        return {
            "id": folder.id,
            "path": folder.path,
            "folder_path": folder.path,
            "kind": folder.kind,
            "status": folder.status,
            "model_name": folder.model_name,
            "embed_dim": folder.embed_dim,
            "total_files": folder.files_total,
            "embedded_files": folder.files_embedded,
            "failed_files": folder.files_failed,
            "skipped_files": folder.files_skipped,
            "auto_embed_new": folder.auto_embed_new,
            "is_available": folder.is_available,
            "last_scanned_at": folder.last_scanned_at,
            "last_indexed_at": folder.last_embedded_at,
            "created_at": folder.created_at,
            "updated_at": folder.updated_at,
        }

    @staticmethod
    def _file_dict(row: EmbeddedFiles) -> dict[str, Any]:
        """Convert a file ORM row into a plain dict."""
        return {
            "id": row.id,
            "embedded_folder_id": row.embedded_folder_id,
            "path": row.path,
            "name": row.name,
            "extension": row.extension,
            "kind": row.kind,
            "mime_type": row.mime_type,
            "size_bytes": row.size_bytes,
            "content_hash": row.content_hash,
            "file_modified_at": row.file_modified_at,
            "status": row.status,
            "error_message": row.error_message,
            "attempt_count": row.attempt_count,
            "model_name": row.model_name,
            "char_count": row.char_count,
            "chunks_count": row.chunks_count,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }

    @staticmethod
    def _chunk_dict_to_stored(row: DocumentChunk, *, with_vectors: bool = True) -> StoredChunk:
        """Convert a chunk ORM row into a :class:`StoredChunk`."""
        vector = None
        if with_vectors and row.vector is not None and row.has_vector:
            vector = [float(value) for value in row.vector]
        return StoredChunk(
            id=row.id,
            source_type=row.source_type,
            source_path=row.source_path,
            source_name=row.source_name,
            mime_type=row.mime_type,
            chunk_index=row.chunk_index,
            text=row.text,
            char_count=row.char_count,
            page_number=row.page_number,
            section_label=row.section_label,
            model_name=row.model_name,
            embed_dim=row.embed_dim,
            vector=vector,
            has_vector=bool(row.has_vector),
            embedded_file_id=row.embedded_file_id,
            article_id=row.article_id,
        )

    @staticmethod
    def _image_to_stored(row: ImageEmbeddingRecord, *, with_vectors: bool = True) -> StoredImage:
        """Convert an image ORM row into a :class:`StoredImage`."""
        vector = None
        if with_vectors and row.vector is not None:
            vector = [float(value) for value in row.vector]
        return StoredImage(
            id=row.id,
            path=row.path,
            name=row.name,
            mime_type=row.mime_type,
            model_name=row.model_name,
            embed_dim=row.embed_dim,
            vector=vector,
            caption=row.caption,
            embedded_file_id=row.embedded_file_id,
        )


knowledge_store = KnowledgeStore()
