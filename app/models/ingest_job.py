"""Persistence model for asynchronous indexing jobs."""

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Enum,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.utils.database import Base


class JobStatus(str, enum.Enum):
    """Execution state of a background job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, enum.Enum):
    """Supported background job kinds."""

    EMBED_FILE = "embed_file"
    EMBED_FOLDER = "embed_folder"
    REINDEX_FOLDER = "reindex_folder"
    REINDEX_ALL = "reindex_all"
    INDEX_ARTICLE = "index_article"


class IngestJob(Base):
    """Row-per-job record used to expose queue progress and history via the API."""

    __tablename__ = "ingest_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    job_type: Mapped[JobType] = mapped_column(
        Enum(
            JobType,
            native_enum=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
        index=True,
    )
    status: Mapped[JobStatus] = mapped_column(
        Enum(
            JobStatus,
            native_enum=False,
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
        default=JobStatus.QUEUED,
        index=True,
    )

    payload: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB(astext_type=Text()), "postgresql"),
        nullable=False,
        default=dict,
    )

    progress_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    progress_done: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    progress_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    progress_skipped: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    message: Mapped[str | None] = mapped_column(String(500), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        CheckConstraint(
            "progress_done >= 0 AND progress_failed >= 0 AND progress_skipped >= 0",
            name="chk_ingest_jobs_progress_non_negative",
        ),
    )
