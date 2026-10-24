"""SQLAlchemy engine, session factory, Postgres/pgvector probing and schema bootstrap.

The engine is created lazily and every probe is tolerant: the app must import and run even
when no PostgreSQL driver, server or pgvector extension is present (services then fall back to
in-memory knowledge storage). Probe results are cached per process.
"""

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config.settings import settings

logger = logging.getLogger(__name__)


def normalise_database_url(url: str) -> str:
    """Pin bare ``postgresql://`` URLs to the psycopg2 driver.

    SQLAlchemy 2.1 resolves bare postgres URLs to psycopg-3, which this project does not
    install, so an explicit scheme keeps ``DATABASE_URL`` values from docker-compose working.

    Args:
        url: Configured database URL.

    Returns:
        str: URL with an explicit driver scheme.
    """
    if url.startswith("postgresql://"):
        return "postgresql+psycopg2://" + url[len("postgresql://") :]
    return url


DATABASE_URL = normalise_database_url(settings.database_url)

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None
_lock = threading.Lock()
_db_available: bool | None = None
_schema_ready: bool = False


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def get_engine() -> Engine:
    """Return the lazily created SQLAlchemy engine.

    Returns:
        Engine: Engine bound to :data:`DATABASE_URL`.
    """
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the lazily created session factory."""
    global _session_factory
    if _session_factory is None:
        with _lock:
            if _session_factory is None:
                _session_factory = sessionmaker(
                    bind=get_engine(), autocommit=False, autoflush=False
                )
    return _session_factory


class _EngineProxy:
    """Forwards attribute access to the engine created on first use."""

    def __getattr__(self, name: str):
        """Delegate to :func:`get_engine` so ``from app.utils.database import engine`` works."""
        return getattr(get_engine(), name)


engine: Engine = _EngineProxy()  # type: ignore[assignment]
SessionLocal = get_session_factory


def db_is_available() -> bool:
    """Whether Postgres is reachable, caching the first probe result.

    Returns:
        bool: True when a connection and a trivial query succeed.
    """
    global _db_available
    if _db_available is not None:
        return _db_available
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        _db_available = True
    except (SQLAlchemyError, ModuleNotFoundError) as exc:
        logger.warning(
            f"Database unavailable ({exc.__class__.__name__}): using in-memory knowledge fallback"
        )
        _db_available = False
    except Exception as exc:  # noqa: BLE001 - driver/OS level failures must not crash startup
        logger.warning(f"Database unavailable ({exc.__class__.__name__}): degraded mode active")
        _db_available = False
    return _db_available


# Readable alias used by services
database_is_available = db_is_available


def reset_db_availability_probe() -> None:
    """Clear the cached availability probe so services re-check connectivity."""
    global _db_available
    _db_available = None


def enable_pgvector() -> bool:
    """Create the ``vector`` extension when privileges allow.

    Returns:
        bool: True when the extension is present or was created.
    """
    if not db_is_available():
        return False
    try:
        with get_engine().begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        return True
    except SQLAlchemyError as exc:
        logger.warning(f"Could not enable pgvector extension: {exc}")
        return False


def vector_extension_installed() -> bool:
    """Whether the pgvector extension is installed in the current database."""
    if not db_is_available():
        return False
    try:
        with get_engine().connect() as conn:
            return (
                conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).scalar()
                is not None
            )
    except SQLAlchemyError:
        return False


def init_schema() -> bool:
    """Create missing tables from declarative metadata (idempotent bootstrap).

    Alembic stays the source of truth for team workflows; this keeps local demos runnable
    without a migration step.

    Returns:
        bool: True when the schema was created/verified successfully.
    """
    if not db_is_available():
        return False

    import app.models as models  # local import to avoid a circular import at module load

    try:
        if not vector_extension_installed():
            enable_pgvector()
        models.Base.metadata.create_all(bind=get_engine())
        logger.info(
            f"Schema ready. Tables: {', '.join(sorted(inspect(get_engine()).get_table_names()))}"
        )
        return True
    except SQLAlchemyError as exc:
        logger.warning(f"Schema bootstrap failed: {exc}")
        return False


def safe_schema_bootstrap(*, force: bool = False) -> bool:
    """Ensure vector tables exist, enabling pgvector first when possible.

    Args:
        force: Re-run the bootstrap even when it already succeeded.

    Returns:
        bool: True when the vector tables are present.
    """
    global _schema_ready
    if _schema_ready and not force:
        return True
    if not db_is_available():
        return False
    if not enable_pgvector():
        logger.warning(
            "pgvector extension unavailable — vector columns cannot be created; "
            "knowledge storage stays in memory"
        )
        return False
    if not init_schema():
        return False
    _schema_ready = True
    return True


@contextmanager
def get_db_session() -> Iterator[Session]:
    """Yield a session with commit-on-success, rollback-on-error semantics.

    Yields:
        Session: An open SQLAlchemy session bound to the lazily created engine.
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped database session."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def database_status() -> dict:
    """Database and pgvector status for the diagnostics endpoint.

    Returns:
        dict: Connectivity, extension and schema- readiness flags plus the redacted URL.
    """
    url = DATABASE_URL
    if "@" in url:
        scheme, _, rest = url.partition("://")
        user = rest.split("@", 1)[0].split(":", 1)[0]
        host = rest.split("@", 1)[1] if "@" in rest else ""
        url = f"{scheme}://{user}:***@{host}"
    return {
        "url": url,
        "available": db_is_available(),
        "pgvector_installed": vector_extension_installed(),
        "schema_ready": _schema_ready,
    }
