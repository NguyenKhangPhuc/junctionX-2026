"""Main FastAPI application entrypoint.

Initializes middleware and registers API routing for video processing, dubbing, and system healthcheck.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure project root directory is in sys.path when running as a direct script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.articles import router as articles_router
from app.api.videos import router as videos_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("junction-backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown events."""
    logger.info("Application starting up...")
    yield
    logger.info("Application shutting down...")


app = FastAPI(
    title="Junction Multilingual Video & Article Backend",
    description="Backend service providing video audio extraction, STT transcription, translation, and TTS dubbing.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers under /api namespace
app.include_router(videos_router, prefix="/api")
app.include_router(articles_router, prefix="/api")


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Healthcheck endpoint for monitoring service readiness.

    Returns:
        dict[str, str]: Status dictionary indicating service health.
    """
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=5605,
        reload=True,
    )
