import inspect
import json
import logging
import queue
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session, joinedload

from app.models.video import Video
from app.models.video_dubbed_version import VideoDubbedVersion
from app.schemas.video import (
    DubbedDeleteResponse,
    DubbedLanguagesResponse,
    DubbedVersionResponse,
    VideoDetailResponse,
    VideoProcessRequest,
    VideoProcessResponse,
)
from app.services.ffmpeg_service import ffmpeg_service
from app.services.video_pipeline_service import VideoPipelineService, video_pipeline_service
from app.utils.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/videos", tags=["videos"])


@router.post(
    "/process",
    response_model=VideoProcessResponse,
    summary="Process and dub a video",
    description=(
        "Executes the deterministic video pipeline with HITL gate: extracts audio if missing, "
        "transcribes speech, translates transcript to target language using LLM, "
        "and pauses in pending_review for human journalist approval before TTS/muxing."
    ),
)
def process_video_endpoint(
    request: VideoProcessRequest,
    db: Session = Depends(get_db),
) -> VideoProcessResponse:
    """Trigger the video processing and dubbing workflow with HITL review gate."""
    try:
        call_kwargs = dict(
            db=db,
            video_path=request.video_path,
            target_language=request.target_language,
            article_id=request.article_id,
            voice=request.voice,
            tts_engine=request.tts_engine,
        )
        sig = inspect.signature(video_pipeline_service.process_video)
        if "pause_for_review" in sig.parameters:
            call_kwargs["pause_for_review"] = True

        response = video_pipeline_service.process_video(**call_kwargs)
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


@router.post(
    "/process-stream",
    summary="Process and dub a video with real-time SSE progress streaming",
    description="Streams real-time Server-Sent Events (SSE) indicating pipeline progress milestones, synthesis segments, and final VideoProcessResponse.",
)
def process_video_stream_endpoint(
    request: VideoProcessRequest,
    db: Session = Depends(get_db),
):
    """Trigger the video dubbing workflow and stream real-time progress via Server-Sent Events."""
    def event_generator():
        q: queue.Queue = queue.Queue()

        def on_progress(stage: str, percent: float, message: str, extra: Optional[Dict[str, Any]] = None):
            q.put({
                "stage": stage,
                "percent": round(percent, 1),
                "message": message,
                "data": extra,
            })

        def worker():
            from app.utils.database import SessionLocal
            with SessionLocal() as worker_db:
                try:
                    res = video_pipeline_service.process_video(
                        db=worker_db,
                        video_path=request.video_path,
                        target_language=request.target_language,
                        article_id=request.article_id,
                        voice=request.voice,
                        tts_engine=request.tts_engine,
                        pause_for_review=False,
                        progress_callback=on_progress,
                    )
                    q.put({
                        "stage": "completed",
                        "percent": 100.0,
                        "message": "Dubbing pipeline completed successfully!",
                        "data": res.model_dump(mode="json"),
                    })
                except Exception as exc:
                    logger.error(f"Error in dubbing pipeline stream worker: {exc}", exc_info=True)
                    q.put({
                        "stage": "error",
                        "percent": 0.0,
                        "error": str(exc),
                    })
                finally:
                    q.put(None)  # Sentinel

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            try:
                item = q.get(timeout=1.0)
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"
            except queue.Empty:
                # SSE keep-alive comment every second during long operations
                yield ": keep-alive\n\n"

        thread.join(timeout=5.0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
    """Retrieve a single video by ID."""
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


@router.post(
    "",
    response_model=VideoDetailResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and register a new video",
    description="Uploads a video file, saves it to /data/videos/raw/, probes its duration, and creates a database record.",
)
def create_video_endpoint(
    file: UploadFile = File(..., description="Target video media file to upload"),
    article_id: Optional[int] = Form(None, description="Optional article ID to associate with this video"),
    original_language: str = Form("vi", description="Source spoken language code"),
    db: Session = Depends(get_db),
) -> VideoDetailResponse:
    """Upload a raw video file and register a Video entity."""
    try:
        # 1. Determine destination path on disk
        safe_filename = Path(file.filename).name
        dest_dir = Path(__file__).resolve().parent.parent.parent / "data" / "videos" / "raw"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / safe_filename

        # 2. Write uploaded chunks to disk
        with dest_file.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Saved uploaded video file to disk: {dest_file}")

        # 3. Probe video duration
        duration = ffmpeg_service.get_media_duration(str(dest_file))

        # 4. Format path as requested: /data/videos/raw/...
        stored_path = f"/data/videos/raw/{safe_filename}"

        # 5. Persist Video entity
        video = Video(
            file_name=safe_filename,
            file_path=stored_path,
            duration=duration,
            original_language=original_language,
            article_id=article_id,
            is_default_file=True,
        )
        db.add(video)
        db.commit()
        db.refresh(video)

        logger.info(f"Registered video ID {video.id} with path {stored_path} (duration: {duration:.2f}s).")
        return VideoDetailResponse.model_validate(video)
    except Exception as exc:
        db.rollback()
        logger.error(f"Failed to upload video: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video upload failed: {exc}",
        ) from exc


@router.get(
    "/{video_id}/dubbed",
    response_model=DubbedLanguagesResponse,
    summary="Get all dubbed versions of a video",
    description="Retrieve all dubbed language versions and supported language codes for a given video.",
)
def get_video_dubbed_versions(
    video_id: int,
    db: Session = Depends(get_db),
) -> DubbedLanguagesResponse:
    """Retrieve all dubbed versions and supported language codes for a video."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if video is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with ID {video_id} not found.",
        )

    dubbed_versions = (
        db.query(VideoDubbedVersion)
        .filter(VideoDubbedVersion.video_id == video_id)
        .order_by(VideoDubbedVersion.created_at.asc())
        .all()
    )

    languages = list(dict.fromkeys([d.target_language for d in dubbed_versions]))
    return DubbedLanguagesResponse(
        video_id=video.id,
        languages=languages,
        dubbed_versions=[DubbedVersionResponse.model_validate(d) for d in dubbed_versions],
    )


@router.delete(
    "/{video_id}/dubbed/{target_language}",
    response_model=DubbedDeleteResponse,
    summary="Delete a dubbed video version",
    description="Deletes a specific dubbed version by target language, removing database records and deleting physical video and audio files from disk.",
)
def delete_video_dubbed_version(
    video_id: int,
    target_language: str,
    db: Session = Depends(get_db),
) -> DubbedDeleteResponse:
    """Delete a dubbed version and its underlying media files."""
    video = db.query(Video).filter(Video.id == video_id).first()
    if video is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video with ID {video_id} not found.",
        )

    dubbed_version = (
        db.query(VideoDubbedVersion)
        .filter(
            VideoDubbedVersion.video_id == video_id,
            VideoDubbedVersion.target_language == target_language,
        )
        .first()
    )
    if dubbed_version is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dubbed version for language '{target_language}' not found for video ID {video_id}.",
        )

    deleted_files = []

    # 1. Delete dubbed video file
    if dubbed_version.dubbed_video_path:
        resolved_video_file = VideoPipelineService.resolve_media_path(dubbed_version.dubbed_video_path)
        if resolved_video_file.exists() and resolved_video_file.is_file():
            resolved_video_file.unlink(missing_ok=True)
            deleted_files.append(str(resolved_video_file))
            logger.info(f"Deleted dubbed video file: {resolved_video_file}")

    # 2. Delete dubbed audio file
    if dubbed_version.dubbed_audio_path:
        resolved_audio_file = VideoPipelineService.resolve_media_path(dubbed_version.dubbed_audio_path)
        if resolved_audio_file.exists() and resolved_audio_file.is_file():
            resolved_audio_file.unlink(missing_ok=True)
            deleted_files.append(str(resolved_audio_file))
            logger.info(f"Deleted dubbed audio file: {resolved_audio_file}")

    # 3. Delete temporary clips folder if present
    clips_dir = ffmpeg_service.dubbed_audio_dir / f"clips_{video_id}_{target_language}"
    if clips_dir.exists() and clips_dir.is_dir():
        shutil.rmtree(clips_dir, ignore_errors=True)
        deleted_files.append(str(clips_dir))
        logger.info(f"Deleted temporary clips folder: {clips_dir}")

    # 4. Remove database record
    db.delete(dubbed_version)
    db.commit()

    logger.info(
        f"Dubbed version for language '{target_language}' of video ID {video_id} deleted successfully."
    )
    return DubbedDeleteResponse(
        video_id=video_id,
        target_language=target_language,
        message=f"Dubbed version for language '{target_language}' and associated media files deleted successfully.",
        deleted_files=deleted_files,
    )