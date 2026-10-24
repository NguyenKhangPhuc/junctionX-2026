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
        tts_engine: str = "xtts",
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
        need_transcribe = (
            not video.original_transcript_json
            or not video.original_transcript_text
            or (
                isinstance(video.original_transcript_json, list)
                and len(video.original_transcript_json) > 0
                and float(video.original_transcript_json[0].get("start", 0.0)) == 0.0
            )
        )

        if need_transcribe:
            logger.info(f"Transcribing original audio with VAD filter for video ID {video.id}...")
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

        # Translate transcript if needed (or if re-transcribed with VAD)
        need_translate = (
            need_transcribe
            or not dubbed_version.transcript_json
            or not dubbed_version.transcript_text
            or (
                isinstance(dubbed_version.transcript_json, list)
                and len(dubbed_version.transcript_json) > 0
                and float(dubbed_version.transcript_json[0].get("start", 0.0)) == 0.0
            )
        )

        if need_translate:
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
            dubbed_version.dubbed_audio_path = None
            dubbed_version.dubbed_video_path = None
            db.commit()
            db.refresh(dubbed_version)
        else:
            logger.info(
                f"Translated transcript already cached for video ID {video.id} in {target_language}."
            )

        # If no spoken speech was detected at all, handle gracefully without crashing TTS
        if not (dubbed_version.transcript_text or "").strip():
            logger.info(
                f"No speech detected in video ID {video.id}. Keeping original audio and video."
            )
            dubbed_version.dubbed_audio_path = video.original_audio_path
            dubbed_version.dubbed_video_path = video.file_path
            dubbed_version.status = "completed"
            db.commit()
            db.refresh(dubbed_version)
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

        dubbed_audio_dest = (
            ffmpeg_service.dubbed_audio_dir
            / f"{video.id}_{target_language}_{resolved_video.stem}.wav"
        )

        audio_dur_mismatch = False
        if (
            dubbed_version.dubbed_audio_path
            and Path(dubbed_version.dubbed_audio_path).exists()
            and (video.duration or 0.0) > 2.0
        ):
            current_audio_dur = ffmpeg_service.get_media_duration(dubbed_version.dubbed_audio_path)
            if abs(current_audio_dur - video.duration) > 1.0:
                logger.info(
                    f"Dubbed audio duration ({current_audio_dur:.2f}s) mismatch with video ({video.duration:.2f}s). Re-synthesizing..."
                )
                audio_dur_mismatch = True

        need_tts = (
            need_translate
            or audio_dur_mismatch
            or not dubbed_version.dubbed_audio_path
            or not Path(dubbed_version.dubbed_audio_path).exists()
            or str(dubbed_audio_dest) != str(Path(dubbed_version.dubbed_audio_path).resolve())
            or (tts_engine.lower() == "xtts" and "_edge" in str(dubbed_version.dubbed_audio_path))
        )

        if need_tts:
            logger.info(
                f"Synthesizing dubbed speech for video ID {video.id} in {target_language} (engine: {tts_engine})..."
            )
            # Extract clean speaker sample from original audio if voice cloning is active
            speaker_sample_path = None
            if video.original_audio_path:
                try:
                    start_offset = 0.0
                    if video.original_transcript_json and isinstance(video.original_transcript_json, list):
                        for seg in video.original_transcript_json:
                            if isinstance(seg, dict) and seg.get("text", "").strip():
                                start_offset = float(seg.get("start", 0.0))
                                break
                    speaker_sample_path = ffmpeg_service.extract_speaker_sample(
                        video.original_audio_path,
                        start_time=start_offset,
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not extract speaker sample ({e})."
                    )

            # Estimate speaker pitch and gender to inform TTS voice selection
            detected_gender = "male"
            ref_for_pitch = speaker_sample_path or video.original_audio_path
            if ref_for_pitch and Path(ref_for_pitch).exists():
                detected_gender = tts_service.estimate_gender(ref_for_pitch)
                logger.info(
                    f"Estimated speaker gender for video ID {video.id}: '{detected_gender}'"
                )

            segments_to_synth = dubbed_version.transcript_json
            has_segments = (
                isinstance(segments_to_synth, list)
                and len(segments_to_synth) > 0
                and any(s.get("text", "").strip() for s in segments_to_synth if isinstance(s, dict))
            )

            if has_segments:
                logger.info(
                    f"Synthesizing {len(segments_to_synth)} segments on timeline for video ID {video.id}..."
                )
                clips_dir = ffmpeg_service.dubbed_audio_dir / f"clips_{video.id}_{target_language}"
                clips = tts_service.synthesize_segments(
                    segments=segments_to_synth,
                    target_language=target_language,
                    output_dir=clips_dir,
                    speaker_wav_path=speaker_sample_path,
                    voice=voice,
                    engine=tts_engine,
                    gender=detected_gender,
                )
                total_media_dur = video.duration or ffmpeg_service.get_media_duration(str(resolved_video))
                dubbed_audio_path = ffmpeg_service.build_timeline_track(
                    clips=clips,
                    total_duration=total_media_dur,
                    output_path=str(dubbed_audio_dest),
                    sample_rate=24000,
                    max_speedup=1.35,
                )
            else:
                logger.info(
                    f"No structured segments found for video ID {video.id}. Synthesizing full text directly..."
                )
                dubbed_audio_path = tts_service.synthesize_speech(
                    text=dubbed_version.transcript_text or "",
                    target_language=target_language,
                    speaker_wav_path=speaker_sample_path,
                    output_path=str(dubbed_audio_dest),
                    voice=voice,
                    engine=tts_engine,
                    gender=detected_gender,
                )

            dubbed_version.dubbed_audio_path = dubbed_audio_path
            dubbed_version.dubbed_video_path = None
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
            or need_tts
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
