"""Video model representing original source videos and default audio/transcript tracks."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.utils.database import Base

if TYPE_CHECKING:
    from app.models.article import Article
    from app.models.video_dubbed_version import VideoDubbedVersion


class Video(Base):
    """Video entity storing metadata, default original audio, and transcription."""

    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("articles.id", ondelete="SET NULL"), nullable=True, index=True
    )
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    original_audio_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    duration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    original_language: Mapped[str] = mapped_column(String(10), default="vi", nullable=False)
    original_transcript_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    original_transcript_json: Mapped[Optional[Any]] = mapped_column(JSON().with_variant(JSONB, "postgresql"), nullable=True)
    is_default_file: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    article: Mapped[Optional["Article"]] = relationship("Article", back_populates="videos")
    dubbed_versions: Mapped[List["VideoDubbedVersion"]] = relationship(
        "VideoDubbedVersion", back_populates="video", cascade="all, delete-orphan"
    )
