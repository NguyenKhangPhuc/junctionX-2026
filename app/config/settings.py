"""Centralized application settings loaded from environment variables and `.env`.

Everything Phase 2 (RAG indexing, hybrid search, multi-agent orchestration) reads its
configuration from here so that model names, vector dimensions and retrieval knobs can be
tuned without touching service code.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Runtime configuration for the Junction backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ core infra
    app_name: str = "Junction Multilingual Video & Article Backend"
    app_data_dir: Path = PROJECT_ROOT / "data"
    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5433/vectordb"
    redis_url: str | None = None  # reserved for a future distributed job queue
    api_token: str | None = None
    allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # ------------------------------------------------------------------ LLM backends
    ollama_host: str = "http://127.0.0.1:11434"
    llm_model: str = "llama3.2"
    openrouter_api_key: str | None = None
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    llm_timeout_seconds: float = 90.0
    llm_temperature: float = 0.2
    # Set to false to skip local Ollama probing/calls entirely (e.g. hosted-only mode)
    ollama_enabled: bool = True
    openrouter_base_url: str = "https://openrouter.io/api/v1"
    # Output token cap for agent generations (completions, not context window)
    llm_max_tokens: int = 4096
    # Ollama context window: keep modest so VRAM stays free for embedding models
    ollama_num_ctx: int = 8192
    # Structured-output parse retries before an agent step is marked failed
    json_retry_attempts: int = 3

    # ------------------------------------------------------------------ embeddings
    # Multilingual text encoder (Vietnamese + English) used for document chunks/articles.
    text_embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    text_embed_dim: int = 384
    # CLIP-style encoder used for image indexing and text->image search.
    image_embed_model: str = "jinaai/jina-clip-v1"
    image_embed_dim: int = 768
    rerank_model: str = "BAAI/bge-reranker-v2-m3-int8"
    embed_batch_size: int = 64
    embed_cache_dir: str | None = None
    # ONNX runtime optimisation hints for the embedding/rerank models
    max_output_tokens: int = 4096
    text_embed_max_retries: int = 2
    embedding_download_timeout_seconds: float = 300.0

    # ------------------------------------------------------------------ chunking
    chunk_size: int = 900  # characters per chunk
    chunk_overlap: int = 150
    max_document_bytes: int = 25 * 1024 * 1024  # 25 MB per file

    # ------------------------------------------------------------------ hybrid retrieval
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    vector_weight: float = 0.6
    lexical_weight: float = 0.4
    rrf_k: int = 60
    rerank_candidate_factor: int = 3
    rerank_enabled: bool = True
    max_evidence_chunks: int = 8
    lexical_index_ttl_seconds: float = 60.0
    # Per-leg candidate depth before RRF fusion
    hybrid_candidate_limit: int = 40
    # Final number of evidence hits returned to the UI / agents
    hybrid_final_k: int = 6
    # Dense hits below this cosine similarity are dropped
    vector_min_score: float = 0.2
    # pgvector cosine distance ceiling (1 - similarity) for candidate filtering
    vector_max_distance: float = 0.8
    rerank_top_k: int = 12

    # ------------------------------------------------------------------ agents
    agent_max_revisions: int = 2
    agent_max_retrieval_rounds: int = 2
    agent_evidence_char_limit: int = 600
    # Run the fact-checker/citation validator pass before publishing an article
    article_verify_enabled: bool = True

    # ------------------------------------------------------------------ job queue
    job_worker_count: int = 2

    @property
    def upload_dir(self) -> Path:
        """Directory where uploaded documents and images are persisted."""
        return self.app_data_dir / "uploads"

    @property
    def ollama_base_url(self) -> str:
        """Ollama API base URL."""
        return self.ollama_host.rstrip("/")

    @property
    def ollama_model(self) -> str:
        """Active local chat model name."""
        return self.llm_model

    @property
    def reranker_model(self) -> str:
        """Cross-encoder reranker model name."""
        return self.rerank_model

    @property
    def max_file_bytes(self) -> int:
        """Per-file ingestion size limit."""
        return self.max_document_bytes

    @property
    def lexical_refresh_seconds(self) -> float:
        """BM25 corpus rebuild TTL in seconds."""
        return self.lexical_index_ttl_seconds

    @property
    def index_worker_workers(self) -> int:
        """Indexing job worker count."""
        return self.job_worker_count

    @property
    def dubbing_dir(self) -> Path:
        """Directory where dubbed video/audio artifacts are written."""
        return self.app_data_dir / "dubbing"

    def ensure_data_dirs(self) -> None:
        """Create the on-disk data directories used by uploads and media artifacts."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.dubbing_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance (single source of truth per process)."""
    settings = Settings()
    settings.ensure_data_dirs()
    return settings


settings = get_settings()
