"""Hybrid retrieval: dense vector search + BM25 lexical search + RRF fusion + rerank."""

import logging
import time
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from app.config.settings import settings
from app.services.bm25_service import BM25Index, bm25_index
from app.services.embedding_service import (
    EmbeddingDimensionError,
    EmbeddingUnavailableError,
    embedding_service,
)
from app.services.knowledge_store import KnowledgeStore, StoredChunk, StoredImage, knowledge_store

logger = logging.getLogger(__name__)

HitKind = Literal["document", "article", "image"]

_CHUNK_PREFIX = "chunk:"
_IMAGE_PREFIX = "image:"


@dataclass
class SearchHit:
    """A fused retrieval hit with traceable vector, lexical and rerank scores."""

    kind: HitKind
    path: str
    name: str
    text: str = ""
    snippet: str = ""
    marker: str = ""
    source_type: str = "file"
    chunk_id: str | None = None
    image_id: str | None = None
    chunk_index: int | None = None
    page: int | None = None
    section: str | None = None
    vector_score: float | None = None
    lexical_score: float | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None
    model_name: str | None = None
    caption: str | None = None

    @property
    def score(self) -> float:
        """Best available score: rerank when present, otherwise fused RRF score."""
        return self.rerank_score if self.rerank_score is not None else self.fused_score

    def citation(self) -> dict[str, Any]:
        """Citation record suitable for ``Article.citations``."""
        if self.kind == "image":
            return {
                "marker": self.marker,
                "image_id": self.image_id,
                "source_type": "image",
                "path": self.path,
                "name": self.name,
                "caption": self.caption,
                "scores": self.scores(),
            }
        return {
            "marker": self.marker,
            "chunk_id": self.chunk_id,
            "source_type": self.source_type,
            "path": self.path,
            "name": self.name,
            "page": self.page,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "scores": self.scores(),
        }

    def scores(self) -> dict[str, float]:
        """Score breakdown for the debug trace panel."""
        scores = {"fused": round(self.fused_score, 6)}
        if self.vector_score is not None:
            scores["vector"] = round(self.vector_score, 4)
        if self.lexical_score is not None:
            scores["lexical"] = round(self.lexical_score, 4)
        if self.rerank_score is not None:
            scores["rerank"] = round(self.rerank_score, 4)
        return scores

    def to_dict(self) -> dict[str, Any]:
        """API-facing representation (excerpt text only, no full chunk payload)."""
        return {
            "marker": self.marker,
            "kind": self.kind,
            "source_type": self.source_type,
            "path": self.path,
            "name": self.name,
            "snippet": self.snippet,
            "page": self.page,
            "section": self.section,
            "chunk_id": self.chunk_id,
            "image_id": self.image_id,
            "chunk_index": self.chunk_index,
            "model": self.model_name,
            "caption": self.caption,
            "score": round(self.score, 6),
            "scores": self.scores(),
        }


@dataclass
class _Candidate:
    """Internal candidate keyed by a lexical corpus id."""

    corpus_id: str
    kind: HitKind
    chunk: StoredChunk | None = None
    image: StoredImage | None = None
    vector_score: float | None = None
    lexical_score: float | None = None
    vector_rank: int | None = None
    lexical_rank: int | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None

    @property
    def text(self) -> str:
        """Text used for rerank and evidence building."""
        if self.chunk is not None:
            return self.chunk.text
        if self.image is not None:
            return self.image.caption or self.image.name
        return ""


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    k: int = 60,
) -> dict[str, float]:
    """Fuse multiple ranked id lists with Reciprocal Rank Fusion.

    Args:
        ranked_lists: Ordered lists of corpus ids, best first.
        k: RRF constant (default 60) damping the influence of deep ranks.

    Returns:
        dict[str, float]: Fused score per corpus id.
    """
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, corpus_id in enumerate(ranked, start=1):
            fused[corpus_id] = fused.get(corpus_id, 0.0) + 1.0 / (k + rank)
    return fused


class HybridSearchService:
    """Combines pgvector similarity, BM25 lexical ranking and cross-encoder reranking."""

    def __init__(
        self,
        store: KnowledgeStore | None = None,
        embeddings: Any | None = None,
        lexical_index: BM25Index | None = None,
    ) -> None:
        self.store = store or knowledge_store
        self.embeddings = embeddings or embedding_service
        self.lexical = lexical_index or bm25_index
        self._lexical_ready = False
        self._lexical_built_at = 0.0
        self._built_from_chunks = 0
        self._built_from_images = 0

    # ------------------------------------------------------------------ lexical corpus
    def invalidate_lexical_index(self) -> None:
        """Mark the BM25 corpus stale so the next search rebuilds it."""
        self._lexical_ready = False

    def ensure_lexical_index(self, *, force: bool = False) -> int:
        """Build or refresh the BM25 corpus from the knowledge store.

        Args:
            force: Rebuild even when the cached corpus is considered fresh.

        Returns:
            int: Number of indexed documents.
        """
        ttl = settings.lexical_refresh_seconds
        fresh = self._lexical_ready and (time.monotonic() - self._lexical_built_at) < ttl
        if fresh and not force:
            return len(self.lexical)

        documents: list[tuple[str, str]] = []
        chunks = self.store.iter_chunks(with_vectors=False)
        for chunk in chunks:
            header = chunk.section_label or chunk.source_name
            documents.append(
                (f"{_CHUNK_PREFIX}{chunk.id}", f"{header}\n{chunk.text}" if header else chunk.text)
            )
        images = self.store.iter_images(with_vectors=False)
        for image in images:
            documents.append(
                (
                    f"{_IMAGE_PREFIX}{image.id}",
                    f"{image.caption or ''} {image.name} image".strip(),
                )
            )

        self.lexical.clear()
        self.lexical.add_many(documents)
        self._lexical_ready = True
        self._lexical_built_at = time.monotonic()
        self._built_from_chunks = len(chunks)
        self._built_from_images = len(images)
        logger.debug(
            f"Rebuilt BM25 corpus: {len(chunks)} chunks, {len(images)} images"
        )
        return len(documents)

    # ------------------------------------------------------------------ vector legs
    def _vector_text_hits(
        self, query: str, limit: int, kinds: set[HitKind]
    ) -> list[tuple[StoredChunk, float]]:
        """Dense text search restricted to the requested kinds."""
        if "document" not in kinds and "article" not in kinds:
            return []
        if not self.embeddings.text_available():
            return []
        source_types = []
        if "document" in kinds:
            source_types.append("file")
        if "article" in kinds:
            source_types.append("article")
        try:
            vector = self.embeddings.embed_query(query)
        except (EmbeddingUnavailableError, EmbeddingDimensionError) as exc:
            logger.warning(f"Query embedding failed; lexical-only retrieval: {exc}")
            return []
        try:
            hits = self.store.vector_search_chunks(
                vector,
                max(1, limit),
                source_types=source_types or None,
                max_distance=settings.vector_max_distance,
            )
        except Exception as exc:  # noqa: BLE001 - SQL/type errors surface as retrieval misses
            logger.warning(f"Vector chunk search failed: {exc}")
            return []
        return [(chunk, score) for chunk, score in hits if score >= settings.vector_min_score]

    def _vector_image_hits(
        self, query: str, limit: int, kinds: set[HitKind]
    ) -> list[tuple[StoredImage, float]]:
        """Dense CLIP text->image search."""
        if "image" not in kinds or not self.embeddings.image_available():
            return []
        try:
            vector = self.embeddings.embed_image_query(query)
        except (EmbeddingUnavailableError, EmbeddingDimensionError) as exc:
            logger.warning(f"CLIP query embedding failed: {exc}")
            return []
        try:
            hits = self.store.vector_search_images(
                vector, max(1, limit), max_distance=settings.vector_max_distance
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Vector image search failed: {exc}")
            return []
        return [(image, score) for image, score in hits if score >= settings.vector_min_score]

    def _lexical_hits(self, query: str, limit: int) -> list[tuple[str, float]]:
        """BM25 ranking over the chunk/image corpus."""
        if not self.lexical.is_empty:
            return self.lexical.search(query, max(1, limit))
        return []

    # ------------------------------------------------------------------ assembly
    @staticmethod
    def _snippet(text: str, limit: int = 320) -> str:
        """Shorten text for UI previews without cutting mid-word mid-sentence."""
        clean = " ".join((text or "").split())
        if len(clean) <= limit:
            return clean
        trimmed = clean[:limit].rsplit(" ", 1)[0]
        return f"{trimmed}..."

    def _candidate_from_corpus_id(self, corpus_id: str) -> _Candidate | None:
        """Resolve a lexical corpus id back to a stored chunk or image."""
        if corpus_id.startswith(_CHUNK_PREFIX):
            chunk = self.store.get_chunk(corpus_id[len(_CHUNK_PREFIX) :])
            if chunk is None:
                return None
            kind: HitKind = "article" if chunk.source_type.value == "article" else "document"
            return _Candidate(corpus_id=corpus_id, kind=kind, chunk=chunk)
        if corpus_id.startswith(_IMAGE_PREFIX):
            image = self.store.get_image(corpus_id[len(_IMAGE_PREFIX) :])
            if image is None:
                return None
            return _Candidate(corpus_id=corpus_id, kind="image", image=image)
        return None

    def search(
        self,
        query: str,
        k: int | None = None,
        *,
        kinds: Sequence[HitKind] = ("document", "article", "image"),
        candidate_limit: int | None = None,
        rerank: bool | None = None,
    ) -> list[SearchHit]:
        """Run hybrid retrieval for a query.

        Args:
            query: Natural-language query (any language).
            k: Number of final hits (defaults to ``settings.hybrid_final_k``).
            kinds: Hit kinds to retrieve.
            candidate_limit: Per-leg candidate depth before fusion.
            rerank: Override the configured rerank flag.

        Returns:
            list[SearchHit]: Ranked hits carrying evidence markers ``E1..En``.
        """
        query = (query or "").strip()
        if not query:
            return []

        final_k = k if k and k > 0 else settings.hybrid_final_k
        depth = candidate_limit or settings.hybrid_candidate_limit
        active_kinds = set(kinds)
        self.ensure_lexical_index()

        vector_chunks = self._vector_text_hits(query, depth, active_kinds)
        vector_images = self._vector_image_hits(query, depth, active_kinds)
        lexical = self._lexical_hits(query, depth * 2)

        candidates: dict[str, _Candidate] = {}
        for rank, (chunk, score) in enumerate(vector_chunks, start=1):
            corpus_id = f"{_CHUNK_PREFIX}{chunk.id}"
            candidate = candidates.setdefault(
                corpus_id,
                _Candidate(
                    corpus_id=corpus_id,
                    kind="article" if chunk.source_type.value == "article" else "document",
                    chunk=chunk,
                ),
            )
            candidate.vector_score = score
            candidate.vector_rank = rank
        for rank, (image, score) in enumerate(vector_images, start=1):
            corpus_id = f"{_IMAGE_PREFIX}{image.id}"
            candidate = candidates.setdefault(
                corpus_id, _Candidate(corpus_id=corpus_id, kind="image", image=image)
            )
            candidate.vector_score = score
            candidate.vector_rank = rank

        lexical_depth = max(1, depth)
        wants_text = "document" in active_kinds or "article" in active_kinds
        for rank, (corpus_id, score) in enumerate(lexical[: lexical_depth * 2], start=1):
            if corpus_id.startswith(_CHUNK_PREFIX) and not wants_text:
                continue
            if corpus_id.startswith(_IMAGE_PREFIX) and "image" not in active_kinds:
                continue
            candidate = candidates.setdefault(
                corpus_id, _Candidate(corpus_id=corpus_id, kind="document")
            )
            candidate.lexical_score = score
            candidate.lexical_rank = rank

        # Resolve lexical-only candidates that were not loaded by the vector leg
        for corpus_id, candidate in candidates.items():
            if candidate.chunk is None and candidate.image is None:
                resolved = self._candidate_from_corpus_id(corpus_id)
                if resolved is None:
                    continue
                candidate.chunk = resolved.chunk
                candidate.image = resolved.image
                if resolved.kind in active_kinds:
                    candidate.kind = resolved.kind
                else:
                    continue

        fused = reciprocal_rank_fusion(
            [
                [c.corpus_id for c in sorted(candidates.values(), key=lambda c: (c.vector_rank or 10_000)) if c.vector_rank],
                [c.corpus_id for c in sorted(candidates.values(), key=lambda c: (c.lexical_rank or 10_000)) if c.lexical_rank],
            ],
            k=settings.rrf_k,
        )
        for corpus_id, score in fused.items():
            candidates[corpus_id].fused_score = score

        usable = [
            candidate
            for candidate in candidates.values()
            if candidate.kind in active_kinds and (candidate.chunk or candidate.image)
        ]
        usable.sort(key=lambda candidate: candidate.fused_score, reverse=True)
        pool = usable[: max(final_k * 2, final_k)]

        should_rerank = settings.rerank_enabled if rerank is None else rerank
        if should_rerank and pool and settings.rerank_top_k:
            self._apply_rerank(query, pool)

        pool.sort(key=lambda candidate: candidate.rerank_score if candidate.rerank_score is not None else candidate.fused_score, reverse=True)

        hits: list[SearchHit] = []
        for index, candidate in enumerate(pool[:final_k], start=1):
            hits.append(self._to_hit(candidate, marker=f"E{index}"))
        return hits

    def _apply_rerank(self, query: str, candidates: list[_Candidate]) -> None:
        """Score candidates with the cross-encoder, tolerating reranker failures."""
        top_k = min(len(candidates), settings.rerank_top_k)
        subset = candidates[:top_k]
        if not self.embeddings.rerank_available():
            return
        try:
            scores = self.embeddings.rerank(query, [candidate.text for candidate in subset])
        except Exception as exc:  # noqa: BLE001 - rerank is best-effort
            logger.warning(f"Rerank skipped: {exc}")
            return
        for candidate, score in zip(subset, scores):
            candidate.rerank_score = float(score)

    def _to_hit(self, candidate: _Candidate, *, marker: str) -> SearchHit:
        """Build a public :class:`SearchHit` from an internal candidate."""
        if candidate.chunk is not None:
            chunk = candidate.chunk
            return SearchHit(
                kind=candidate.kind,
                marker=marker,
                source_type=chunk.source_type.value,
                path=chunk.source_path,
                name=chunk.source_name,
                text=chunk.text,
                snippet=self._snippet(chunk.text),
                chunk_id=str(chunk.id),
                chunk_index=chunk.chunk_index,
                page=chunk.page_number,
                section=chunk.section_label,
                vector_score=candidate.vector_score,
                lexical_score=candidate.lexical_score,
                fused_score=candidate.fused_score,
                rerank_score=candidate.rerank_score,
                model_name=chunk.model_name,
            )
        image = candidate.image
        assert image is not None  # filtered by callers
        return SearchHit(
            kind="image",
            marker=marker,
            source_type="image",
            path=image.path,
            name=image.name,
            text=image.caption or image.name,
            snippet=self._snippet(image.caption or image.name, limit=200),
            image_id=str(image.id),
            caption=image.caption,
            vector_score=candidate.vector_score,
            lexical_score=candidate.lexical_score,
            fused_score=candidate.fused_score,
            rerank_score=candidate.rerank_score,
            model_name=image.model_name,
        )

    # ------------------------------------------------------------------ convenience wrappers
    def search_documents(self, query: str, k: int | None = None) -> list[SearchHit]:
        """Text/document retrieval only (no image hits)."""
        return self.search(query, k, kinds=("document", "article"))

    def search_images(self, query: str, k: int | None = None) -> list[SearchHit]:
        """Text-to-image retrieval only."""
        return self.search(query, k, kinds=("image",), rerank=False)

    def evidence_pack(
        self,
        hits: Sequence[SearchHit],
        *,
        per_hit_chars: int = 900,
        total_chars: int = 6000,
    ) -> str:
        """Render hits as an evidence block for LLM prompts.

        Args:
            hits: Retrieval hits in rank order.
            per_hit_chars: Character budget per hit.
            total_chars: Overall character budget.

        Returns:
            str: Formatted evidence block with ``[E#]`` markers and provenance headers.
        """
        blocks: list[str] = []
        used = 0
        for hit in hits:
            body = " ".join((hit.text or "").split())
            if len(body) > per_hit_chars:
                body = body[:per_hit_chars].rsplit(" ", 1)[0] + "..."
            location = f" page {hit.page}" if hit.page else ""
            section = f" section: {hit.section}" if hit.section else ""
            header = f"[{hit.marker}] {hit.name} ({hit.source_type}; path: {hit.path}{location}{section})"
            block = f"{header}\n{body}"
            if used + len(block) > total_chars:
                break
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks)

    def citations(self, hits: Sequence[SearchHit]) -> list[dict[str, Any]]:
        """Citation records for a hit list."""
        return [hit.citation() for hit in hits]

    def stats(self) -> dict[str, Any]:
        """Retrieval backend statistics for the diagnostics endpoint."""
        return {
            "backend": self.store.backend,
            "lexical_documents": len(self.lexical),
            "lexical_indexed_chunks": self._built_from_chunks,
            "lexical_indexed_images": self._built_from_images,
            "lexical_fresh": self._lexical_ready,
            "lexical_stats": self.lexical.stats(),
            "embedding": self.embeddings.status(),
            "settings": {
                "candidate_limit": settings.hybrid_candidate_limit,
                "final_k": settings.hybrid_final_k,
                "rrf_k": settings.rrf_k,
                "rerank_enabled": settings.rerank_enabled,
                "rerank_top_k": settings.rerank_top_k,
                "vector_min_score": settings.vector_min_score,
            },
        }


hybrid_search_service = HybridSearchService()
