"""Orchestration service for the deterministic video processing and dubbing workflow."""

import logging
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.models.video import Video
from app.models.video_dubbed_version import VideoDubbedVersion
from app.schemas.video import DubbedVersionResponse, VideoProcessResponse
from app.services.ffmpeg_service import ffmpeg_service
from app.services.stt_service import stt_service
from app.services.translation_service import translation_service
from app.services.tts_service import tts_service

logger = logging.getLogger(__name__)


class VideoPipelineService:
    """End-to-end pipeline service for processing videos, extracting audio, transcribing, translating, and dubbing."""

    def process_video(
        self,
        db: Session,
        video_path: str,
        target_language: str = "en",
        article_id: Optional[int] = None,
        voice: Optional[str] = None,
    ) -> VideoProcessResponse:
        """Execute the end-to-end video dubbing pipeline.

        1. Verify video exists and register or retrieve Video entity.
        2. Extract original audio (is_default_file=True) using FFmpeg if not already present.
        3. Transcribe speech using faster-whisper (is_default_file=True).
        4. Translate transcript to target_language using LLM (is_default_file=False).
        5. Synthesize translated speech using edge-tts.
        6. Merge translated audio with source video using FFmpeg.

        Args:
            db: Active SQLAlchemy database session.
            video_path: Path to the local video file.
            target_language: ISO language code to translate and dub into.
            article_id: Optional ID of the parent article.
            voice: Optional TTS voice override.

        Returns:
            VideoProcessResponse: Complete details of original and dubbed assets.

        Raises:
            FileNotFoundError: If the source video path does not exist.
            RuntimeError: If any pipeline step fails.
        """
        resolved_video = Path(video_path).resolve()
        if not resolved_video.exists():
            raise FileNotFoundError(f"Input video file not found at: {video_path}")

        # 1. Register or find Video record
        video = (
            db.query(Video)
            .filter(Video.file_path == str(resolved_video))
            .first()
        )

        duration = ffmpeg_service.get_media_duration(str(resolved_video))

        if video is None:
            logger.info(f"Registering new video record for: {resolved_video.name}")
            video = Video(
                file_name=resolved_video.name,
                file_path=str(resolved_video),
                duration=duration,
                article_id=article_id,
                original_language="vi",
                is_default_file=True,
            )
            db.add(video)
            db.commit()
            db.refresh(video)
        else:
            if article_id and video.article_id != article_id:
                video.article_id = article_id
                db.commit()

        # 2. Extract original audio (is_default_file = True)
        need_audio_extraction = (
            not video.original_audio_path
            or not Path(video.original_audio_path).exists()
        )

        if need_audio_extraction:
            logger.info(f"Extracting original default audio for video ID {video.id}...")
            raw_audio_path = ffmpeg_service.extract_audio(str(resolved_video))
            video.original_audio_path = raw_audio_path
            video.is_default_file = True
            db.commit()
            db.refresh(video)
        else:
            logger.info(
                f"Original audio already present for video ID {video.id}: {video.original_audio_path}"
            )

        # 3. Transcribe original speech (is_default_file = True)
        if not video.original_transcript_json or not video.original_transcript_text:
            logger.info(f"Transcribing original audio for video ID {video.id}...")
            stt_result = stt_service.transcribe_audio(video.original_audio_path)
            video.original_transcript_text = stt_result["full_text"]
            video.original_transcript_json = stt_result["segments"]
            video.original_language = stt_result["language"]
            db.commit()
            db.refresh(video)
        else:
            logger.info(f"Original transcript already exists for video ID {video.id}.")

        # 4. Handle Dubbed Version (is_default_file = False)
        dubbed_version = (
            db.query(VideoDubbedVersion)
            .filter(
                VideoDubbedVersion.video_id == video.id,
                VideoDubbedVersion.target_language == target_language,
            )
            .first()
        )

        if dubbed_version is None:
            logger.info(
                f"Creating new dubbed version record for video ID {video.id} (target: {target_language})..."
            )
            dubbed_version = VideoDubbedVersion(
                video_id=video.id,
                target_language=target_language,
                status="processing",
                is_default_file=False,
            )
            db.add(dubbed_version)
            db.commit()
            db.refresh(dubbed_version)

        # Translate transcript if needed
        if not dubbed_version.transcript_json or not dubbed_version.transcript_text:
            logger.info(
                f"Translating transcript for video ID {video.id} from {video.original_language} to {target_language}..."
            )
            trans_text, trans_segments = translation_service.translate_segments(
                segments=video.original_transcript_json or [],
                source_lang=video.original_language,
                target_lang=target_language,
            )
            dubbed_version.transcript_text = trans_text
            dubbed_version.transcript_json = trans_segments
            db.commit()
            db.refresh(dubbed_version)
        else:
            logger.info(
                f"Translated transcript already cached for video ID {video.id} in {target_language}."
            )

        # 5. Synthesize speech for translated transcript
        need_tts = (
            not dubbed_version.dubbed_audio_path
            or not Path(dubbed_version.dubbed_audio_path).exists()
        )

        if need_tts:
            logger.info(
                f"Synthesizing dubbed speech for video ID {video.id} in {target_language}..."
            )
            dubbed_audio_dest = (
                ffmpeg_service.dubbed_audio_dir
                / f"{video.id}_{target_language}_{resolved_video.stem}.mp3"
            )
            dubbed_audio_path = tts_service.synthesize_speech(
                text=dubbed_version.transcript_text or "",
                target_language=target_language,
                output_path=str(dubbed_audio_dest),
                voice=voice,
            )
            dubbed_version.dubbed_audio_path = dubbed_audio_path
            db.commit()
            db.refresh(dubbed_version)
        else:
            logger.info(
                f"Dubbed audio already present for video ID {video.id}: {dubbed_version.dubbed_audio_path}"
            )

        # 6. Merge dubbed audio into video
        need_merge = (
            not dubbed_version.dubbed_video_path
            or not Path(dubbed_version.dubbed_video_path).exists()
        )

        if need_merge:
            logger.info(
                f"Merging dubbed audio into video for video ID {video.id}..."
            )
            dubbed_video_dest = (
                ffmpeg_service.dubbed_video_dir
                / f"{video.id}_{target_language}_{resolved_video.stem}.mp4"
            )
            dubbed_video_path = ffmpeg_service.merge_audio_video(
                video_path=video.file_path,
                audio_path=dubbed_version.dubbed_audio_path,
                output_path=str(dubbed_video_dest),
            )
            dubbed_version.dubbed_video_path = dubbed_video_path
            dubbed_version.status = "completed"
            db.commit()
            db.refresh(dubbed_version)
        else:
            logger.info(
                f"Dubbed video already present for video ID {video.id}: {dubbed_version.dubbed_video_path}"
            )
            dubbed_version.status = "completed"
            db.commit()

        # Build response
        return VideoProcessResponse(
            video_id=video.id,
            file_name=video.file_name,
            file_path=video.file_path,
            original_audio_path=video.original_audio_path,
            duration=video.duration,
            original_language=video.original_language,
            original_transcript_text=video.original_transcript_text,
            original_transcript_json=video.original_transcript_json,
            is_default_file=True,
            dubbed_version=DubbedVersionResponse.model_validate(dubbed_version),
        )


video_pipeline_service = VideoPipelineService()
