"""Article model representing generated, community and published articles."""

import enum
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional

from sqlalchemy import JSON, DateTime, Enum, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.utils.database import Base

if TYPE_CHECKING:
    from app.models.embeddings import DocumentChunk
    from app.models.video import Video


class ArticleStatus(str, enum.Enum):
    """Editorial state of an article produced by the Content Specialist agent."""

    DRAFT = "draft"
    NEEDS_REVIEW = "needs_review"
    VERIFIED = "verified"


class Article(Base):
    """Article entity containing text content, citations, language, and associated media."""

    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    original_language: Mapped[str] = mapped_column(String(10), default="vi", nullable=False)
    author: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Phase 2 bookkeeping: Content Specialist output + Verification Agent verdict
    status: Mapped[ArticleStatus] = mapped_column(
        Enum(
            ArticleStatus,
            native_enum=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
        default=ArticleStatus.DRAFT,
        index=True,
    )
    source_query: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    thread_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)

    # Citation list: [{"marker": "E1", "source_type": "file", "path": ..., "page": ...}]
    citations: Mapped[List[dict[str, Any]]] = mapped_column(
        JSON().with_variant(JSONB(astext_type=Text()), "postgresql"),
        nullable=False,
        default=list,
    )
    verification_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    verification_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    videos: Mapped[List["Video"]] = relationship(
        "Video", back_populates="article", cascade="all, delete-orphan"
    )
    chunks: Mapped[List["DocumentChunk"]] = relationship(
        "DocumentChunk", back_populates="article", cascade="all, delete-orphan"
    )
