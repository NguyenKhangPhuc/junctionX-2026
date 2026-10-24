"""Dense embedding service backed by fastembed (text, CLIP image and cross-encoder reranker).

Model weights are loaded lazily and cached per process, so a request that only needs lexical
retrieval never pays the ONNX startup cost. When ONNX models cannot be downloaded or loaded,
the service reports itself as unavailable and callers fall back to BM25-only retrieval.
"""

import logging
import threading
from typing import Any, Iterable, Sequence

from app.config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingUnavailableError(RuntimeError):
    """Raised when the requested embedding or rerank model cannot be loaded."""


class EmbeddingDimensionError(ValueError):
    """Raised when a model output dimension does not match the configured vector column."""


class EmbeddingService:
    """Lazy wrapper around fastembed text/image encoders and a cross-encoder reranker."""

    def __init__(
        self,
        text_model: str | None = None,
        image_model: str | None = None,
        rerank_model: str | None = None,
        text_dim: int | None = None,
        image_dim: int | None = None,
        cache_dir: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.text_model = text_model or settings.text_embed_model
        self.image_model = image_model or settings.image_embed_model
        self.rerank_model = rerank_model or settings.rerank_model
        self.text_dim = text_dim if text_dim is not None else settings.text_embed_dim
        self.image_dim = image_dim if image_dim is not None else settings.image_embed_dim
        self.cache_dir = cache_dir if cache_dir is not None else settings.embed_cache_dir
        self.batch_size = batch_size if batch_size is not None else settings.embed_batch_size

        self._lock = threading.Lock()
        self._text_encoder: Any | None = None
        self._image_encoder: Any | None = None
        self._image_query_encoder: Any | None = None
        self._reranker: Any | None = None
        self._load_errors: dict[str, str] = {}

    # ------------------------------------------------------------------ lifecycle helpers
    def reset(self) -> None:
        """Drop loaded models and cached load errors (used by tests and reconfiguration)."""
        with self._lock:
            self._text_encoder = None
            self._image_encoder = None
            self._image_query_encoder = None
            self._reranker = None
            self._load_errors = {}

    @staticmethod
    def _supported_names(registry: Iterable[dict[str, Any]]) -> set[str]:
        """Collect model names from a fastembed registry description list."""
        names: set[str] = set()
        for item in registry:
            if isinstance(item, dict):
                name = item.get("model") or item.get("model_name")
                if name:
                    names.add(str(name))
        return names

    def _load_error(self, key: str) -> str | None:
        return self._load_errors.get(key)

    def _remember_error(self, key: str, message: str) -> None:
        self._load_errors[key] = message
        logger.warning(f"{key} model unavailable: {message}")

    def _base_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        return kwargs

    # ------------------------------------------------------------------ availability flags
    def text_available(self) -> bool:
        """True when the dense text encoder is loaded or loadable."""
        return self._ensure_text_encoder(quiet=True) is not None

    def image_available(self) -> bool:
        """True when both CLIP image and CLIP text query encoders are loadable."""
        if not self._ensure_image_encoder(quiet=True):
            return False
        return self._ensure_image_query_encoder(quiet=True) is not None

    def rerank_available(self) -> bool:
        """True when the cross-encoder reranker is loadable."""
        return self._ensure_reranker(quiet=True) is not None

    # ------------------------------------------------------------------ lazy loaders
    def _ensure_text_encoder(self, quiet: bool = False) -> Any | None:
        cached = self._load_error("text")
        if cached is not None:
            if not quiet:
                raise EmbeddingUnavailableError(cached)
            return None
        if self._text_encoder is not None:
            return self._text_encoder

        try:
            from fastembed import TextEmbedding

            with self._lock:
                if self._text_encoder is None:
                    logger.info(f"Loading text embedding model '{self.text_model}'...")
                    self._text_encoder = TextEmbedding(
                        model_name=self.text_model, **self._base_kwargs()
                    )
            return self._text_encoder
        except Exception as exc:  # noqa: BLE001 - download/runtime/registry errors
            message = f"{exc.__class__.__name__}: {exc}"
            self._remember_error("text", message)
            if not quiet:
                raise EmbeddingUnavailableError(message) from exc
            return None

    def _ensure_image_encoder(self, quiet: bool = False) -> Any | None:
        cached = self._load_error("image")
        if cached is not None:
            if not quiet:
                raise EmbeddingUnavailableError(cached)
            return None
        if self._image_encoder is not None:
            return self._image_encoder

        try:
            from fastembed import ImageEmbedding

            with self._lock:
                if self._image_encoder is None:
                    logger.info(f"Loading image embedding model '{self.image_model}'...")
                    self._image_encoder = ImageEmbedding(
                        model_name=self.image_model, **self._base_kwargs()
                    )
            return self._image_encoder
        except Exception as exc:  # noqa: BLE001
            message = f"{exc.__class__.__name__}: {exc}"
            self._remember_error("image", message)
            if not quiet:
                raise EmbeddingUnavailableError(message) from exc
            return None

    def _image_query_model_name(self) -> str:
        """Map a CLIP vision model name to its text-side counterpart."""
        from fastembed import TextEmbedding

        candidates = [self.image_model, self.image_model.replace("-vision", "-text")]
        supported = self._supported_names(TextEmbedding.list_supported_models())
        for candidate in candidates:
            if candidate in supported:
                return candidate
        raise EmbeddingUnavailableError(
            f"No CLIP text encoder found for image model '{self.image_model}'. "
            f"Set IMAGE_EMBED_MODEL to a paired CLIP model such as 'jinaai/jina-clip-v1'."
        )

    def _ensure_image_query_encoder(self, quiet: bool = False) -> Any | None:
        cached = self._load_error("image_query")
        if cached is not None:
            if not quiet:
                raise EmbeddingUnavailableError(cached)
            return None
        if self._image_query_encoder is not None:
            return self._image_query_encoder

        try:
            from fastembed import TextEmbedding

            model_name = self._image_query_model_name()
            with self._lock:
                if self._image_query_encoder is None:
                    logger.info(f"Loading CLIP text query model '{model_name}'...")
                    self._image_query_encoder = TextEmbedding(
                        model_name=model_name, **self._base_kwargs()
                    )
            return self._image_query_encoder
        except Exception as exc:  # noqa: BLE001
            message = f"{exc.__class__.__name__}: {exc}"
            self._remember_error("image_query", message)
            if not quiet:
                raise EmbeddingUnavailableError(message) from exc
            return None

    def _ensure_reranker(self, quiet: bool = False) -> Any | None:
        cached = self._load_error("rerank")
        if cached is not None:
            if not quiet:
                raise EmbeddingUnavailableError(cached)
            return None
        if self._reranker is not None:
            return self._reranker

        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            with self._lock:
                if self._reranker is None:
                    logger.info(f"Loading reranker model '{self.rerank_model}'...")
                    self._reranker = TextCrossEncoder(
                        model_name=self.rerank_model, **self._base_kwargs()
                    )
            return self._reranker
        except Exception as exc:  # noqa: BLE001
            message = f"{exc.__class__.__name__}: {exc}"
            self._remember_error("rerank", message)
            if not quiet:
                raise EmbeddingUnavailableError(message) from exc
            return None

    # ------------------------------------------------------------------ inference
    @staticmethod
    def _to_vector(raw: Any) -> list[float]:
        """Convert a numpy array / list output into a plain float list."""
        if hasattr(raw, "tolist"):
            return [float(value) for value in raw.tolist()]
        return [float(value) for value in raw]

    def _validate_dim(self, vector: Sequence[float], expected: int, component: str) -> list[float]:
        """Ensure model output width matches the pgvector column width."""
        if expected and len(vector) != expected:
            raise EmbeddingDimensionError(
                f"{component} model produced {len(vector)} dimensions but the schema column "
                f"expects {expected}. Update {component.upper()}_EMBED_MODEL / *_EMBED_DIM to match."
            )
        return list(vector)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document/passage texts with the dense text encoder.

        Args:
            texts: Texts to encode.

        Returns:
            list[list[float]]: One vector per input text, in the same order.

        Raises:
            EmbeddingUnavailableError: The text encoder could not be loaded.
            EmbeddingDimensionError: Output dimension mismatches the schema column.
        """
        if not texts:
            return []
        encoder = self._ensure_text_encoder()
        if encoder is None:
            raise EmbeddingUnavailableError(self._load_error("text") or "Text encoder unavailable")

        vectors: list[list[float]] = []
        for start in range(0, len(texts), max(1, self.batch_size)):
            batch = texts[start : start + max(1, self.batch_size)]
            for raw in encoder.embed(batch):
                vectors.append(self._validate_dim(self._to_vector(raw), self.text_dim, "text"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single retrieval query with the dense text encoder."""
        return self.embed_texts([text])[0]

    def embed_image(self, path: str) -> list[float]:
        """Embed one image file with the CLIP/vision encoder.

        Raises:
            EmbeddingUnavailableError: The vision encoder could not be loaded.
            EmbeddingDimensionError: Output dimension mismatches the schema column.
        """
        encoder = self._ensure_image_encoder()
        if encoder is None:
            raise EmbeddingUnavailableError(self._load_error("image") or "Image encoder unavailable")
        results = list(encoder.embed([path]))
        if not results:
            raise EmbeddingUnavailableError(f"Image encoder returned no vector for {path}")
        return self._validate_dim(self._to_vector(results[0]), self.image_dim, "image")

    def embed_image_query(self, text: str) -> list[float]:
        """Embed a text query into the CLIP image vector space for text->image search."""
        encoder = self._ensure_image_query_encoder()
        if encoder is None:
            raise EmbeddingUnavailableError(
                self._load_error("image_query") or "CLIP text query encoder unavailable"
            )
        results = list(encoder.embed([text]))
        if not results:
            raise EmbeddingUnavailableError("CLIP text query encoder returned no vector")
        return self._validate_dim(self._to_vector(results[0]), self.image_dim, "image")

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        """Score passages against a query with the cross-encoder reranker.

        Args:
            query: Retrieval query.
            passages: Candidate passage texts.

        Returns:
            list[float]: Relevance scores (higher is better) aligned with `passages`.

        Raises:
            EmbeddingUnavailableError: The reranker could not be loaded.
        """
        if not passages:
            return []
        reranker = self._ensure_reranker()
        if reranker is None:
            raise EmbeddingUnavailableError(self._load_error("rerank") or "Reranker unavailable")
        scores = list(reranker.rerank(query, list(passages)))
        return [float(score) for score in scores]

    def warmup(self) -> dict[str, bool]:
        """Preload configured models, reporting which components became available.

        Returns:
            dict[str, bool]: Availability flags for text, image and rerank components.
        """
        return {
            "text": self.text_available(),
            "image": self.image_available(),
            "rerank": self.rerank_available(),
        }

    def status(self) -> dict[str, Any]:
        """Describe configured models and their current load state for diagnostics."""
        return {
            "text_model": self.text_model,
            "text_dim": self.text_dim,
            "text_loaded": self._text_encoder is not None,
            "image_model": self.image_model,
            "image_dim": self.image_dim,
            "image_loaded": self._image_encoder is not None,
            "image_query_loaded": self._image_query_encoder is not None,
            "rerank_model": self.rerank_model,
            "rerank_loaded": self._reranker is not None,
            "errors": dict(self._load_errors),
        }


embedding_service = EmbeddingService()
