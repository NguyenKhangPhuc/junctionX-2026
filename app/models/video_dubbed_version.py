"""Video dubbed version model representing translated audio, transcript, and merged video."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.utils.database import Base

if TYPE_CHECKING:
    from app.models.video import Video


class VideoDubbedVersion(Base):
    """Dubbed video entity storing translated transcript, cloned/synthesized audio, and final merged video."""

    __tablename__ = "video_dubbed_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    target_language: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    transcript_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    transcript_json: Mapped[Optional[Any]] = mapped_column(JSON().with_variant(JSONB, "postgresql"), nullable=True)
    dubbed_audio_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    dubbed_video_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    is_default_file: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationship
    video: Mapped["Video"] = relationship("Video", back_populates="dubbed_versions")
