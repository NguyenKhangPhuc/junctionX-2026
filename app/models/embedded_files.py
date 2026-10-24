"""Model tracking individually indexed files inside registered folders."""

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
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

from app.models.embedded_folders import FolderKind
from app.utils.database import Base

if TYPE_CHECKING:
    from app.models.embedded_folders import EmbeddedFolders
    from app.models.embeddings import DocumentChunk, ImageEmbeddingRecord


class FileStatus(str, enum.Enum):
    """Lifecycle state of an indexed file."""

    QUEUED = "queued"
    RUNNING = "running"
    READY = "ready"
    SKIPPED = "skipped"
    FAILED = "failed"
    STALE = "stale"


class EmbeddedFiles(Base):
    """One file (document or image) registered for indexing, with dedupe metadata."""

    __tablename__ = "embedded_files"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Parent folder registration (nullable so single-file embeddings still work)
    embedded_folder_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("embedded_folders.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    kind: Mapped[FolderKind] = mapped_column(
        Enum(
            FolderKind,
            native_enum=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
        index=True,
    )

    path: Mapped[str] = mapped_column(Text, nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    extension: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    mime_type: Mapped[str] = mapped_column(String(120), nullable=False, default="application/octet-stream")
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    # SHA-256 digest of file bytes, used to skip unchanged files on re-scan
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    file_modified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    status: Mapped[FileStatus] = mapped_column(
        Enum(
            FileStatus,
            native_enum=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
        default=FileStatus.QUEUED,
        index=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Indexing results
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunks_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    folder: Mapped["EmbeddedFolders | None"] = relationship(
        "EmbeddedFolders",
        back_populates="files",
    )
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="embedded_file",
        cascade="all, delete-orphan",
    )
    image_embeddings: Mapped[list["ImageEmbeddingRecord"]] = relationship(
        "ImageEmbeddingRecord",
        back_populates="embedded_file",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        CheckConstraint("size_bytes >= 0", name="chk_embedded_files_size_non_negative"),
        CheckConstraint("chunks_count >= 0", name="chk_embedded_files_chunks_non_negative"),
        UniqueConstraint("embedded_folder_id", "path", name="uq_embedded_files_folder_path"),
    )
