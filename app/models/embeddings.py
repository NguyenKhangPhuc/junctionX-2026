"""pgvector-backed storage models for indexed text chunks and image embeddings."""

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from app.config.settings import settings
from app.utils.database import Base

if TYPE_CHECKING:
    from app.models.article import Article
    from app.models.embedded_files import EmbeddedFiles


TEXT_EMBED_DIM = settings.text_embed_dim
IMAGE_EMBED_DIM = settings.image_embed_dim


class SourceType(str, enum.Enum):
    """Origin of a retrievable text chunk."""

    FILE = "file"
    ARTICLE = "article"


class DocumentChunk(Base):
    """Chunked document/article text with its dense vector for semantic retrieval."""

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # A chunk originates either from an indexed file or from a generated/curated article
    embedded_file_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("embedded_files.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    article_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("articles.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    source_type: Mapped[SourceType] = mapped_column(
        Enum(
            SourceType,
            native_enum=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
        default=SourceType.FILE,
        index=True,
    )

    # Human-readable citation target (file path or article title)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    source_name: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(120), nullable=False, default="text/plain")

    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_label: Mapped[str | None] = mapped_column(String(255), nullable=True)

    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    embed_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    vector: Mapped[list[float]] = mapped_column(Vector(TEXT_EMBED_DIM), nullable=False)
    # False when the embedding model was unavailable: text is still stored for BM25 retrieval
    has_vector: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    embedded_file: Mapped["EmbeddedFiles | None"] = relationship(
        "EmbeddedFiles",
        back_populates="chunks",
    )
    article: Mapped["Article | None"] = relationship(
        "Article",
        back_populates="chunks",
    )

    __table_args__ = (
        CheckConstraint("char_count >= 0", name="chk_document_chunks_char_count_non_negative"),
        CheckConstraint("embed_dim > 0", name="chk_document_chunks_embed_dim_positive"),
        UniqueConstraint(
            "embedded_file_id", "chunk_index", name="uq_document_chunks_file_chunk_index"
        ),
        UniqueConstraint(
            "article_id", "chunk_index", name="uq_document_chunks_article_chunk_index"
        ),
    )


class ImageEmbeddingRecord(Base):
    """CLIP vector for an indexed image, searchable with a text query."""

    __tablename__ = "image_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    embedded_file_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("embedded_files.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    path: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(120), nullable=False, default="image/png")
    caption: Mapped[str | None] = mapped_column(Text, nullable=True)

    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    embed_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    vector: Mapped[list[float]] = mapped_column(Vector(IMAGE_EMBED_DIM), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    embedded_file: Mapped["EmbeddedFiles | None"] = relationship(
        "EmbeddedFiles",
        back_populates="image_embeddings",
    )

    __table_args__ = (
        CheckConstraint("embed_dim > 0", name="chk_image_embeddings_embed_dim_positive"),
    )
