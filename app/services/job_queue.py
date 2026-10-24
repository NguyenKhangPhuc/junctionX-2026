"""Async ingestion job queue with ``ingest_jobs`` persistence and progress tracking.

Jobs run in worker tasks, offloading blocking embedding work to threads. When Postgres is
unavailable the queue keeps the same job records in memory so the UI contract is unchanged.
"""

import asyncio
import logging
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from app.config.settings import settings
from app.models.embedded_folders import FolderKind
from app.models.ingest_job import IngestJob, JobStatus, JobType
from app.services.index_service import FolderNotFoundError, PathNotFoundError, index_service
from app.services.knowledge_store import knowledge_store

logger = logging.getLogger(__name__)


class JobCancelledError(RuntimeError):
    """Raised inside a running job when cancellation was requested."""


class JobNotFoundError(KeyError):
    """Raised when a job id is unknown."""


@dataclass
class JobRecord:
    """In-memory job state, mirrored to ``ingest_jobs`` when Postgres is available."""

    id: str
    type: str
    status: str = JobStatus.QUEUED.value
    payload: dict[str, Any] = field(default_factory=dict)
    progress_total: int = 0
    progress_done: int = 0
    progress_failed: int = 0
    progress_skipped: int = 0
    message: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """API-facing job snapshot with percentage progress."""
        total = self.progress_total or 0
        percent = round((self.progress_done / total) * 100, 1) if total else 0.0
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status,
            "payload": self.payload,
            "progress_total": self.progress_total,
            "progress_done": self.progress_done,
            "progress_failed": self.progress_failed,
            "progress_skipped": self.progress_skipped,
            "progress_percent": percent,
            "message": self.message,
            "error": self.error,
            "result": self.result,
            "created_at": _iso(self.created_at),
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
        }


def _iso(value: datetime | None) -> str | None:
    """ISO-8601 string or None."""
    return value.isoformat() if value else None


class JobQueue:
    """Bounded asyncio worker pool executing indexing jobs."""

    def __init__(self, service: Any | None = None, *, workers: int | None = None) -> None:
        self.service = service or index_service
        self.store = knowledge_store
        self.workers_count = max(1, workers or settings.index_worker_workers)
        self._jobs: dict[str, JobRecord] = {}
        self._order: list[str] = []
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers: list[asyncio.Task[Any]] = []
        self._lock = threading.Lock()
        self._cancel_events: dict[str, threading.Event] = {}
        self._db_ids: dict[str, int] = {}
        self._stopping = False

    # ------------------------------------------------------------------ persistence mirror
    def _mirror_create(self, record: JobRecord) -> None:
        """Insert the job row when Postgres is reachable."""
        if not self.store.uses_postgres():
            return
        try:
            with self.store.session() as session:
                row = IngestJob(
                    type=JobType(record.type),
                    status=JobStatus(record.status),
                    payload=record.payload,
                    progress_total=record.progress_total,
                    created_at=record.created_at,
                )
                session.add(row)
                session.flush()
                self._db_ids[record.id] = int(row.id)
        except Exception as exc:  # noqa: BLE001 - queue must keep working without Postgres
            logger.warning(f"Job mirror create failed: {exc}")

    def _mirror_update(self, record: JobRecord) -> None:
        """Update the mirrored job row, tolerating an absent database."""
        job_row_id = self._db_ids.get(record.id)
        if job_row_id is None or not self.store.uses_postgres():
            return
        try:
            with self.store.session() as session:
                row = session.get(IngestJob, job_row_id)
                if row is None:
                    return
                row.status = JobStatus(record.status)
                row.progress_total = record.progress_total
                row.progress_done = record.progress_done
                row.progress_failed = record.progress_failed
                row.progress_skipped = record.progress_skipped
                row.message = record.message
                row.error = record.error
                row.started_at = record.started_at
                row.finished_at = record.finished_at
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Job mirror update failed: {exc}")

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        """Start worker tasks (idempotent)."""
        if self._workers or self._stopping:
            return
        for index in range(self.workers_count):
            self._workers.append(asyncio.create_task(self._worker(index)))
        logger.info(f"Job queue started with {self.workers_count} worker(s)")

    async def stop(self) -> None:
        """Drain the queue and stop worker tasks."""
        self._stopping = True
        for task in self._workers:
            task.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def _worker(self, index: int) -> None:
        """Consume job ids until cancelled."""
        while True:
            job_id = await self._queue.get()
            try:
                record = self._jobs.get(job_id)
                if record is None:
                    continue
                if record.status in (JobStatus.CANCELLED.value, JobStatus.SUCCEEDED.value):
                    continue
                await self._run(record)
            except asyncio.CancelledError:
                self._queue.task_done()
                raise
            except Exception as exc:  # noqa: BLE001 - workers must survive job failures
                logger.exception(f"Job worker {index} crashed on job {job_id}: {exc}")
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------ public API
    async def submit(self, type: JobType | str, payload: dict[str, Any] | None = None) -> JobRecord:
        """Enqueue a job.

        Args:
            type: Job type (``embed_folder``, ``reindex_all``, ...).
            payload: Job payload (paths, folder id, article body, force flag).

        Returns:
            JobRecord: The queued record (progress fields start at zero).
        """
        job_type = JobType(type) if isinstance(type, str) else type
        record = JobRecord(id=str(uuid.uuid4()), type=job_type.value, payload=dict(payload or {}))
        with self._lock:
            self._jobs[record.id] = record
            self._order.append(record.id)
            self._cancel_events[record.id] = threading.Event()
        self._mirror_create(record)
        await self._queue.put(record.id)
        await self.start()
        return record

    def get(self, job_id: str) -> JobRecord | None:
        """Fetch a job record by id."""
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, *, status: JobStatus | str | None = None, limit: int = 50) -> list[JobRecord]:
        """List jobs newest first, optionally filtered by status."""
        wanted = JobStatus(status).value if isinstance(status, str) and status else (
            status.value if status else None
        )
        with self._lock:
            records = [self._jobs[job_id] for job_id in reversed(self._order) if job_id in self._jobs]
        return [record for record in records if wanted is None or record.status == wanted][:limit]

    async def cancel(self, job_id: str) -> JobRecord:
        """Request cancellation of a queued or running job.

        Args:
            job_id: Job id.

        Returns:
            JobRecord: Updated record (status cancelled).

        Raises:
            JobNotFoundError: Unknown job id.
        """
        with self._lock:
            record = self._jobs.get(job_id)
            event = self._cancel_events.get(job_id)
        if record is None:
            raise JobNotFoundError(f"Unknown job id: {job_id}")
        if event:
            event.set()
        if record.status in (JobStatus.QUEUED.value, JobStatus.RUNNING.value):
            record.status = JobStatus.CANCELLED.value
            record.message = "Cancelled"
            record.finished_at = datetime.now(timezone.utc)
            self._mirror_update(record)
        return record

    def stats(self) -> dict[str, Any]:
        """Queue counters for the diagnostics endpoint."""
        with self._lock:
            counts: dict[str, int] = {}
            for record in self._jobs.values():
                counts[record.status] = counts.get(record.status, 0) + 1
            total = len(self._jobs)
        return {
            "jobs": total,
            "by_status": counts,
            "workers": self.workers_count,
            "queued_now": self._queue.qsize(),
            "mirrored_to_db": bool(self._db_ids),
        }

    # ------------------------------------------------------------------ execution
    def _progress_callback(self, record: JobRecord, event: threading.Event) -> Callable[[int, int, str], None]:
        """Build a progress callback that reports file-level progress and honours cancellation."""

        def progress(done: int, total: int, message: str) -> None:
            if event.is_set():
                raise JobCancelledError("Job cancelled")
            record.progress_done = done
            record.progress_total = total or record.progress_total
            record.message = message[:300]
            self._mirror_update(record)

        return progress

    async def _run(self, record: JobRecord) -> None:
        """Execute one job in a worker thread and record its outcome."""
        record.status = JobStatus.RUNNING.value
        record.started_at = datetime.now(timezone.utc)
        record.error = None
        self._mirror_update(record)

        event = self._cancel_events.setdefault(record.id, threading.Event())
        callback = self._progress_callback(record, event)

        try:
            result = await asyncio.to_thread(self._dispatch, record, callback, event)
            record.result = result
            if result:
                record.progress_total = int(result.get("total_files") or record.progress_total)
                record.progress_done = int(result.get("processed", record.progress_done))
                record.progress_failed = int(result.get("failed") or record.progress_failed)
                record.progress_skipped = int(result.get("skipped") or record.progress_skipped)
            record.status = JobStatus.SUCCEEDED.value
            record.message = record.message or "Completed"
        except JobCancelledError as exc:
            record.status = JobStatus.CANCELLED.value
            record.message = "Cancelled"
        except (FolderNotFoundError, PathNotFoundError, KeyError, ValueError) as exc:
            record.status = JobStatus.FAILED.value
            record.error = str(exc)[:1000]
            record.message = "Failed"
            logger.warning(f"Job {record.id} ({record.type}) failed: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface unexpected failures as job errors
            record.status = JobStatus.FAILED.value
            record.error = f"{exc.__class__.__name__}: {exc}"[:1000]
            record.message = "Failed"
            logger.exception(f"Job {record.id} ({record.type}) crashed")
        finally:
            record.finished_at = datetime.now(timezone.utc)
            self._mirror_update(record)

    def _dispatch(
        self, record: JobRecord, callback: Callable[[int, int, str], None], event: threading.Event
    ) -> dict[str, Any]:
        """Map a job type to the corresponding index-service call."""
        payload = record.payload
        job_type = JobType(record.type)

        if job_type is JobType.EMBED_FILE:
            path = payload.get("path") or payload.get("file_path")
            if not path:
                raise ValueError("embed_file job requires a 'path'")
            result = self.service.index_file(
                path,
                kind=payload.get("kind"),
                embedded_folder_id=payload.get("folder_id"),
                force=bool(payload.get("force", False)),
                progress_cb=callback,
            )
            stats = {
                "total_files": 1,
                "processed": 1 if result.status == "ready" else 0,
                "skipped": 1 if result.status == "skipped" else 0,
                "failed": 1 if result.status == "failed" else 0,
                "chunks": result.chunks,
                "images": result.images,
                "chars": result.chars,
                "file_id": result.file_id,
                "status": result.status,
                "detail": result.error or result.skipped_reason,
            }
            return stats

        if job_type is JobType.EMBED_FOLDER:
            folder_id = payload.get("folder_id")
            if not folder_id:
                folder = self.service.register_folder(
                    payload.get("path", ""),
                    FolderKind(payload.get("kind", FolderKind.DOCUMENTS.value)),
                )
                folder_id = str(folder["id"])
                record.payload["folder_id"] = folder_id
            stats = self.service.index_folder(str(folder_id), progress_cb=callback)
            return stats.to_dict()

        if job_type is JobType.REINDEX_FOLDER:
            folder_id = str(payload.get("folder_id") or "")
            if not folder_id:
                raise ValueError("reindex_folder job requires a 'folder_id'")
            stats = self.service.index_folder(folder_id, force=True, progress_cb=callback)
            return stats.to_dict()

        if job_type is JobType.REINDEX_ALL:
            return self.service.index_all(force=bool(payload.get("force", True)), progress_cb=callback)

        if job_type is JobType.INDEX_ARTICLE:
            article_id = int(payload.get("article_id") or 0)
            if not article_id:
                raise ValueError("index_article job requires an 'article_id'")
            return self.service.index_article(
                article_id,
                str(payload.get("title") or f"Article {article_id}"),
                str(payload.get("content") or ""),
            )

        raise ValueError(f"Unsupported job type: {record.type}")


job_queue = JobQueue()
