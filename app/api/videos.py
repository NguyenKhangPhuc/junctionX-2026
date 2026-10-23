"""API router for video processing, dubbing pipeline, and metadata retrieval."""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, joinedload

from app.models.video import Video
from app.schemas.video import (
    VideoDetailResponse,
    VideoProcessRequest,
    VideoProcessResponse,
)
from app.services.video_pipeline_service import video_pipeline_service
from app.utils.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/videos", tags=["videos"])


@router.post(
    "/process",
    response_model=VideoProcessResponse,
    summary="Process and dub a video",
    description=(
        "Executes the deterministic video pipeline: extracts audio if missing (is_default_file=True), "
        "transcribes speech, translates transcript to target language using LLM (is_default_file=False), "
        "synthesizes dubbed audio via TTS, and merges with original video using FFmpeg."
    ),
)
def process_video_endpoint(
    request: VideoProcessRequest,
    db: Session = Depends(get_db),
) -> VideoProcessResponse:
    """Trigger the video processing and dubbing workflow.

    Args:
        request: VideoProcessRequest containing video_path and target_language.
        db: Database session.

    Returns:
        VideoProcessResponse: Complete status and paths of original and dubbed files.

    Raises:
        HTTPException: 404 if video file is missing, 400/500 if processing fails.
    """
    try:
        response = video_pipeline_service.process_video(
            db=db,
            video_path=request.video_path,
            target_language=request.target_language,
            article_id=request.article_id,
            voice=request.voice,
            tts_engine=request.tts_engine,
        )
        return response
    except FileNotFoundError as fnf_err:
        logger.error(f"Video file not found: {fnf_err}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(fnf_err),
        ) from fnf_err
    except ValueError as val_err:
        logger.error(f"Invalid input data: {val_err}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        ) from val_err
    except Exception as exc:
        logger.error(f"Failed to process video: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video pipeline failed: {exc}",
        ) from exc


@router.get(
    "",
    response_model=List[VideoDetailResponse],
    summary="List all videos",
    description="Retrieve all registered videos along with their dubbed versions.",
)
def list_videos(
    db: Session = Depends(get_db),
) -> List[VideoDetailResponse]:
    """Retrieve all videos with attached dubbed versions."""
    videos = (
        db.query(Video)
        .options(joinedload(Video.dubbed_versions))
        .order_by(Video.created_at.desc())
        .all()
    )
    return [VideoDetailResponse.model_validate(v) for v in videos]


@router.get(
    "/{video_id}",
    response_model=VideoDetailResponse,
    summary="Get video details",
    description="Retrieve detailed information for a specific video including its transcripts and dubbed versions.",
)
def get_video_by_id(
    video_id: int,
    db: Session = Depends(get_db),
) -> VideoDetailResponse:
    """Retrieve a single video by ID.

    Args:
        video_id: Primary key ID of the video.
        db: Database session.

    Returns:
        VideoDetailResponse: Video details and associated dubbed versions.

    Raises:
        HTTPException: 404 if video is not found.
    """
    video = (
        db.query(Video)
        .options(joinedload(Video.dubbed_versions))
        .filter(Video.id == video_id)
        .first()
    )
    if video is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with ID {video_id} not found.",
        )

    return VideoDetailResponse.model_validate(video)
