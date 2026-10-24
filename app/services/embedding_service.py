"""Embedding service for text vector representation using FastEmbed."""

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service wrapping FastEmbed to generate dense vector embeddings for text."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        embed_dim: int = 1024,
    ):
        self.model_name = (
            model_name
            or os.getenv("TEXT_EMBED_MODEL")
            or "intfloat/multilingual-e5-large"
        )
        self.embed_dim = int(os.getenv("TEXT_EMBED_DIM", str(embed_dim)))
        self._model = None

    def _get_model(self):
        """Lazily load the FastEmbed TextEmbedding model."""
        if self._model is None:
            from fastembed import TextEmbedding

            # Map bge-m3 alias or typo to true multilingual model
            model_to_load = self.model_name
            if model_to_load in {"bge-m3", "BAAI/bge-m3", "intfloat/multilingua-e5-large", "multilingua-e5-large"}:
                model_to_load = "intfloat/multilingual-e5-large"

            logger.info(f"Loading FastEmbed multilingual model '{model_to_load}' (dim: {self.embed_dim})...")
            self._model = TextEmbedding(model_name=model_to_load)
            logger.info(f"FastEmbed multilingual model '{model_to_load}' initialized successfully.")
        return self._model

    def generate_embedding(self, text: str, is_query: bool = False) -> Optional[List[float]]:
        """Generate a dense 1024-dimensional vector embedding for the input text.

        Args:
            text: Text string to embed.
            is_query: Flag indicating if this text is a search query (formats with query: prefix for E5).

        Returns:
            Optional[List[float]]: Vector representation with length embed_dim, or None on failure.
        """
        clean_text = text.strip()
        if not clean_text:
            return None

        # Apply E5 prefix convention if applicable
        if "e5" in str(self.model_name).lower():
            if is_query and not clean_text.startswith("query:"):
                clean_text = f"query: {clean_text}"
            elif not is_query and not clean_text.startswith("passage:"):
                clean_text = f"passage: {clean_text}"

        try:
            model = self._get_model()
            embeddings = list(model.embed([clean_text]))
            if embeddings:
                vector = [float(v) for v in embeddings[0]]
                return vector
            return None
        except Exception as e:
            logger.warning(f"Could not generate text embedding ({e}). Continuing without embedding.")
            return None


embedding_service = EmbeddingService()
