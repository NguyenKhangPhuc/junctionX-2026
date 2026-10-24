"""Orchestration service for the deterministic video processing, HITL review, and dubbing workflow."""

import logging
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.models.article import Article
from app.models.video import Video
from app.models.video_dubbed_version import VideoDubbedVersion
from app.schemas.video import DubbedVersionResponse, VideoProcessResponse
from app.services.ffmpeg_service import ffmpeg_service
from app.services.stt_service import stt_service
from app.services.translation_service import translation_service
from app.services.tts_service import tts_service
from app.utils.database import SessionLocal

logger = logging.getLogger(__name__)


class VideoPipelineService:
    """End-to-end pipeline service for processing videos, HITL newsroom drafts, and TTS/FFmpeg rendering."""

    def __init__(self, db: Optional[Session] = None):
        """Allow optional session injection for background task compatibility."""
        self.db = db

    @staticmethod
    def resolve_media_path(path_str: str) -> Path:
        """Resolve a media path string, supporting absolute, project-relative, and root-formatted paths."""
        p = Path(path_str)
        if p.exists():
            return p.resolve()
        repo_root = Path(__file__).resolve().parent.parent.parent
        root_rel = (repo_root / path_str.lstrip("/")).resolve()
        if root_rel.exists():
            return root_rel
        cwd_rel = Path(path_str.lstrip("/")).resolve()
        if cwd_rel.exists():
            return cwd_rel
        return root_rel

    @staticmethod
    def _extract_segments_from_transcript_json(
        transcript_json: Any,
    ) -> List[Dict[str, Any]]:
        """Extract time-coded narration segments whether transcript_json is a list or an explanatory dict."""
        if isinstance(transcript_json, list):
            return [s for s in transcript_json if isinstance(s, dict)]
        if isinstance(transcript_json, dict):
            segs = transcript_json.get("spoken_narration_segments")
            if isinstance(segs, list):
                return [s for s in segs if isinstance(s, dict)]
        return []

    @staticmethod
    def _ensure_fallback_video_file(target_path: Path, duration: float = 20.0) -> Path:
        """Generate a minimal 720p broadcast slate MP4 via FFmpeg if an RSS/community draft has no raw video."""
        if target_path.exists():
            return target_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        dur = max(float(duration or 20.0), 5.0)
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x111827:s=1280x720:d={dur:.2f}:r=24",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r=24000:cl=mono:d={dur:.2f}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(target_path),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"Generated fallback broadcast slate video at {target_path}")
        except Exception as e:
            logger.warning(f"Could not generate fallback slate video via FFmpeg ({e}).")
        return target_path

    # =========================================================================
    # PHASE 2 HITL GATE: STEP 1 - 4 (Prepare Draft -> status = "pending_review")
    # =========================================================================
    def prepare_draft(
        self,
        db_or_video_id: Optional[Union[Session, int]] = None,
        video_id: Optional[Union[int, str]] = None,
        video_path: Optional[str] = None,
        target_language: str = "en",
        article_id: Optional[int] = None,
        use_explanatory_prompt: bool = True,
        db: Optional[Session] = None,
    ) -> VideoDubbedVersion:
        """Run Steps 1-4 (Audio Extraction, STT, Explanatory LLM Translation) and pause at 'pending_review'.

        Supports both calling conventions:
          - prepare_draft(video_id, target_language="en")
          - prepare_draft(db=session, video_id=..., target_language="en")
        """
        owns_session = False
        session: Optional[Session] = db or self.db
        resolved_video_id: Optional[int] = None

        if isinstance(db_or_video_id, Session):
            session = db_or_video_id
            if isinstance(video_id, int):
                resolved_video_id = video_id
        elif isinstance(db_or_video_id, int):
            resolved_video_id = db_or_video_id
            if isinstance(video_id, str):
                target_language = video_id
        elif isinstance(video_id, int):
            resolved_video_id = video_id

        if session is None:
            session = SessionLocal()
            owns_session = True

        try:
            video: Optional[Video] = None
            if resolved_video_id is not None:
                video = session.query(Video).filter(Video.id == resolved_video_id).first()
                if not video:
                    raise ValueError(f"Video ID {resolved_video_id} not found.")
                resolved_video = self.resolve_media_path(video.file_path)
            elif video_path is not None:
                resolved_video = self.resolve_media_path(video_path)
                if not resolved_video.exists():
                    raise FileNotFoundError(f"Input video file not found at: {video_path}")
                video = (
                    session.query(Video)
                    .filter(
                        (Video.file_path == str(resolved_video))
                        | (Video.file_path == video_path)
                        | (Video.file_path == f"/data/videos/raw/{resolved_video.name}")
                    )
                    .first()
                )
                if video is None:
                    duration = ffmpeg_service.get_media_duration(str(resolved_video))
                    video = Video(
                        file_name=resolved_video.name,
                        file_path=str(resolved_video),
                        duration=duration,
                        article_id=article_id,
                        original_language="fi",
                        is_default_file=True,
                    )
                    session.add(video)
                    session.commit()
                    session.refresh(video)
            else:
                raise ValueError("Either video_id or video_path must be provided.")

            # Step 2 & 3: Extract audio & transcribe if physical video file exists
            if resolved_video.exists():
                if not video.original_audio_path or not Path(video.original_audio_path).exists():
                    logger.info(f"[HITL Step 2] Extracting original audio for video ID {video.id}...")
                    video.original_audio_path = ffmpeg_service.extract_audio(str(resolved_video))
                    session.commit()
                    session.refresh(video)

                if not video.original_transcript_text or not video.original_transcript_json:
                    logger.info(f"[HITL Step 3] Transcribing audio with Whisper STT for video ID {video.id}...")
                    stt_result = stt_service.transcribe_audio(video.original_audio_path)
                    video.original_transcript_text = stt_result["full_text"]
                    video.original_transcript_json = stt_result["segments"]
                    video.original_language = stt_result["language"]
                    session.commit()
                    session.refresh(video)

            # Pull linked Article context if available
            ouka_context = ""
            source_text = video.original_transcript_text or ""
            if video.article_id:
                article = session.query(Article).filter(Article.id == video.article_id).first()
                if article:
                    source_text = f"{article.title}\n{article.content}\n{source_text}".strip()
                    if "Oulu City Data:" in (article.content or ""):
                        ouka_context = article.content.split("Oulu City Data:", 1)[-1].strip()

            # Step 4: Create or update VideoDubbedVersion with status = "pending_review"
            dubbed_version = (
                session.query(VideoDubbedVersion)
                .filter(
                    VideoDubbedVersion.video_id == video.id,
                    VideoDubbedVersion.target_language == target_language,
                )
                .first()
            )
            if dubbed_version is None:
                dubbed_version = VideoDubbedVersion(
                    video_id=video.id,
                    target_language=target_language,
                    status="pending_review",
                    is_default_file=False,
                )
                session.add(dubbed_version)
                session.commit()
                session.refresh(dubbed_version)

            logger.info(f"[HITL Step 4] Generating explanatory translation draft for video ID {video.id}...")
            if use_explanatory_prompt:
                trans_text, explanatory_json = translation_service.generate_oulu_explanatory_digest(
                    finnish_text=source_text,
                    ouka_context=ouka_context,
                    segments=video.original_transcript_json if isinstance(video.original_transcript_json, list) else None,
                    target_lang=target_language,
                )
                dubbed_version.transcript_text = trans_text
                dubbed_version.transcript_json = explanatory_json
            else:
                trans_text, trans_segments = translation_service.translate_segments(
                    segments=video.original_transcript_json or [],
                    source_lang=video.original_language or "fi",
                    target_lang=target_language,
                )
                dubbed_version.transcript_text = trans_text
                dubbed_version.transcript_json = trans_segments

            # STOP HERE for Human-in-the-Loop review
            dubbed_version.status = "pending_review"
            dubbed_version.dubbed_audio_path = None
            dubbed_version.dubbed_video_path = None
            session.commit()
            session.refresh(dubbed_version)

            logger.info(
                f"[HITL Gate] Draft ID {dubbed_version.id} saved with status='pending_review'. Awaiting human editor approval."
            )
            return dubbed_version
        finally:
            if owns_session:
                session.close()

    # =========================================================================
    # PHASE 2 HITL GATE: STEP 5 - 6 (Approve & Publish -> TTS + FFmpeg Merge)
    # =========================================================================
    def approve_and_publish(
        self,
        dubbed_version_id: int,
        edited_transcript_text: Optional[str] = None,
        edited_transcript_json: Optional[Any] = None,
        db: Optional[Session] = None,
        voice: str = "en-US-GuyNeural",
        tts_engine: str = "edge",
    ) -> VideoDubbedVersion:
        """Apply human editor changes and execute Step 5 (Edge-TTS) and Step 6 (FFmpeg muxing)."""
        owns_session = False
        session = db or self.db
        if session is None:
            session = SessionLocal()
            owns_session = True

        try:
            dub = (
                session.query(VideoDubbedVersion)
                .filter(VideoDubbedVersion.id == dubbed_version_id)
                .first()
            )
            if not dub:
                raise ValueError(f"VideoDubbedVersion ID {dubbed_version_id} not found.")

            video = session.query(Video).filter(Video.id == dub.video_id).first()
            if not video:
                raise ValueError(f"Parent Video ID {dub.video_id} not found.")

            # Apply editor modifications if provided
            if edited_transcript_text is not None:
                dub.transcript_text = edited_transcript_text.strip()
                if isinstance(dub.transcript_json, dict):
                    updated_dict = dict(dub.transcript_json)
                    segs = updated_dict.get("spoken_narration_segments")
                    if isinstance(segs, list) and len(segs) > 0 and isinstance(segs[0], dict):
                        segs[0]["text"] = dub.transcript_text
                        for extra_seg in segs[1:]:
                            if isinstance(extra_seg, dict):
                                extra_seg["text"] = ""
                    else:
                        updated_dict["spoken_narration_segments"] = [
                            {"start": 0.0, "end": video.duration or 15.0, "text": dub.transcript_text}
                        ]
                    dub.transcript_json = updated_dict
                    flag_modified(dub, "transcript_json")

            if edited_transcript_json is not None:
                dub.transcript_json = edited_transcript_json
                flag_modified(dub, "transcript_json")

            dub.status = "processing_media"
            session.commit()
            session.refresh(dub)

            resolved_video = self.resolve_media_path(video.file_path)
            video_stem = resolved_video.stem if resolved_video else f"video_{video.id}"

            # Step 5: Synthesize speech with Edge-TTS (en-US-GuyNeural)
            ffmpeg_service.dubbed_audio_dir.mkdir(parents=True, exist_ok=True)
            dubbed_audio_dest = (
                ffmpeg_service.dubbed_audio_dir
                / f"{video.id}_{dub.target_language}_{video_stem}.wav"
            )
            logger.info(
                f"[HITL Step 5] Synthesizing approved script for dub ID {dub.id} with {tts_engine} ({voice})..."
            )

            segments_to_synth = self._extract_segments_from_transcript_json(dub.transcript_json)
            has_valid_segments = (
                len(segments_to_synth) > 0
                and any(s.get("text", "").strip() for s in segments_to_synth)
                and edited_transcript_text is None
            )

            if has_valid_segments:
                clips_dir = (
                    ffmpeg_service.dubbed_audio_dir
                    / f"clips_{video.id}_{dub.target_language}"
                )
                clips = tts_service.synthesize_segments(
                    segments=segments_to_synth,
                    target_language=dub.target_language,
                    output_dir=clips_dir,
                    speaker_wav_path=None,
                    voice=voice,
                    engine=tts_engine,
                    gender="male",
                )
                total_dur = video.duration or (
                    ffmpeg_service.get_media_duration(str(resolved_video))
                    if resolved_video.exists()
                    else 20.0
                )
                dubbed_audio_path = ffmpeg_service.build_timeline_track(
                    clips=clips,
                    total_duration=total_dur,
                    output_path=str(dubbed_audio_dest),
                    sample_rate=24000,
                    max_speedup=1.35,
                )
            else:
                dubbed_audio_path = tts_service.synthesize_speech(
                    text=dub.transcript_text or "",
                    target_language=dub.target_language,
                    speaker_wav_path=None,
                    output_path=str(dubbed_audio_dest),
                    voice=voice,
                    engine=tts_engine,
                    gender="male",
                )

            dub.dubbed_audio_path = dubbed_audio_path
            session.commit()
            session.refresh(dub)

            # Step 6: Merge dubbed audio into video (generate fallback slate if RSS draft has no physical video)
            if not resolved_video.exists():
                audio_dur = (
                    ffmpeg_service.get_media_duration(dub.dubbed_audio_path)
                    if dub.dubbed_audio_path and Path(dub.dubbed_audio_path).exists()
                    else (video.duration or 20.0)
                )
                self._ensure_fallback_video_file(resolved_video, duration=audio_dur)

            if resolved_video.exists():
                logger.info(
                    f"[HITL Step 6] Merging synthesized audio into video for dub ID {dub.id}..."
                )
                ffmpeg_service.dubbed_video_dir.mkdir(parents=True, exist_ok=True)
                dubbed_video_dest = (
                    ffmpeg_service.dubbed_video_dir
                    / f"{video.id}_{dub.target_language}_{video_stem}.mp4"
                )
                dubbed_video_path = ffmpeg_service.merge_audio_video(
                    video_path=str(resolved_video),
                    audio_path=dub.dubbed_audio_path,
                    output_path=str(dubbed_video_dest),
                )
                dub.dubbed_video_path = dubbed_video_path
            else:
                logger.info(
                    f"[HITL Step 6] Physical video not present at {resolved_video}; audio digest completed at {dub.dubbed_audio_path}."
                )

            dub.status = "completed"
            session.commit()
            session.refresh(dub)
            logger.info(f"[HITL Complete] Dub ID {dub.id} published with status='completed'.")
            return dub

        except Exception as e:
            logger.error(f"Failed to render approved dub ID {dubbed_version_id}: {e}", exc_info=True)
            dub_err = (
                session.query(VideoDubbedVersion)
                .filter(VideoDubbedVersion.id == dubbed_version_id)
                .first()
            )
            if dub_err:
                dub_err.status = "failed"
                session.commit()
            raise
        finally:
            if owns_session:
                session.close()

    def render_approved_dub(self, dub_id: int) -> VideoDubbedVersion:
        """BackgroundTask entrypoint called by /api/newsroom/review/{dub_id}/approve."""
        with SessionLocal() as bg_db:
            return self.approve_and_publish(
                dubbed_version_id=dub_id,
                db=bg_db,
                voice="en-US-GuyNeural",
                tts_engine="edge",
            )

    # =========================================================================
    # PIPELINE ENTRYPOINT (Defaults to HITL pause_for_review=True)
    # =========================================================================
    def process_video(
        self,
        db: Session,
        video_path: str,
        target_language: str = "en",
        article_id: Optional[int] = None,
        voice: Optional[str] = None,
        tts_engine: str = "xtts",
        pause_for_review: bool = True,
        progress_callback: Optional[Callable[[str, float, str, Optional[Dict[str, Any]]], None]] = None,
    ) -> VideoProcessResponse:
        """Execute the end-to-end video dubbing pipeline (pauses at 'pending_review' by default for HITL compliance)."""
        def emit(stage: str, percent: float, msg: str, extra: Optional[Dict[str, Any]] = None):
            if progress_callback:
                try:
                    progress_callback(stage, percent, msg, extra)
                except Exception as cb_err:
                    logger.debug(f"Progress callback exception: {cb_err}")

        emit("init", 5.0, "Validating video file and checking media records...")
        resolved_video = self.resolve_media_path(video_path)
        if not resolved_video.exists():
            raise FileNotFoundError(f"Input video file not found at: {video_path}")

        if pause_for_review:
            dubbed_version = self.prepare_draft(
                db=db,
                video_path=video_path,
                target_language=target_language,
                article_id=article_id,
                use_explanatory_prompt=True,
            )
            video = db.query(Video).filter(Video.id == dubbed_version.video_id).first()
            emit("pending_review", 100.0, "Draft prepared and awaiting human review.", {"video_id": video.id})
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

        # Direct non-HITL execution (when called with pause_for_review=False)
        video = (
            db.query(Video)
            .filter(
                (Video.file_path == str(resolved_video))
                | (Video.file_path == video_path)
                | (Video.file_path == f"/data/videos/raw/{resolved_video.name}")
            )
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

        need_audio_extraction = (
            not video.original_audio_path
            or not Path(video.original_audio_path).exists()
        )

        if need_audio_extraction:
            emit("audio", 12.0, "Extracting audio track from media (FFmpeg)...")
            logger.info(f"Extracting original default audio for video ID {video.id}...")
            raw_audio_path = ffmpeg_service.extract_audio(str(resolved_video))
            video.original_audio_path = raw_audio_path
            video.is_default_file = True
            db.commit()
            db.refresh(video)
            emit("audio", 22.0, "Audio track extracted successfully.")
        else:
            logger.info(
                f"Original audio already present for video ID {video.id}: {video.original_audio_path}"
            )
            emit("audio", 22.0, "Original audio track already present.")

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
            emit("stt", 30.0, "Transcribing speech into time-coded segments (Whisper STT)...")
            logger.info(
                f"Transcribing original audio with VAD filter for video ID {video.id}..."
            )
            stt_result = stt_service.transcribe_audio(video.original_audio_path)
            video.original_transcript_text = stt_result["full_text"]
            video.original_transcript_json = stt_result["segments"]
            video.original_language = stt_result["language"]
            db.commit()
            db.refresh(video)
            seg_count = len(stt_result.get("segments") or [])
            emit("stt", 45.0, f"Speech transcribed successfully ({seg_count} dialogue segments).")
        else:
            logger.info(f"Original transcript already exists for video ID {video.id}.")
            emit("stt", 45.0, "Speech transcript already indexed.")

        dubbed_version = (
            db.query(VideoDubbedVersion)
            .filter(
                VideoDubbedVersion.video_id == video.id,
                VideoDubbedVersion.target_language == target_language,
            )
            .first()
        )

        if dubbed_version is None:
            dubbed_version = VideoDubbedVersion(
                video_id=video.id,
                target_language=target_language,
                status="processing",
                is_default_file=False,
            )
            db.add(dubbed_version)
            db.commit()
            db.refresh(dubbed_version)

        existing_segments = self._extract_segments_from_transcript_json(
            dubbed_version.transcript_json
        )
        need_translate = (
            need_transcribe
            or not dubbed_version.transcript_json
            or not dubbed_version.transcript_text
            or (
                len(existing_segments) > 0
                and float(existing_segments[0].get("start", 0.0)) == 0.0
            )
        )

        if need_translate:
            emit("translate", 50.0, f"Translating transcript into {target_language.upper()} via LLM...")
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
            emit("translate", 62.0, f"Transcript translated into {target_language.upper()} successfully.")
        else:
            logger.info(
                f"Translated transcript already cached for video ID {video.id} in {target_language}."
            )
            emit("translate", 62.0, f"Translated transcript already cached for {target_language.upper()}.")

        if not (dubbed_version.transcript_text or "").strip():
            dubbed_version.dubbed_audio_path = video.original_audio_path
            dubbed_version.dubbed_video_path = video.file_path
            dubbed_version.status = "completed"
            db.commit()
            db.refresh(dubbed_version)
            emit("completed", 100.0, "No speech dialogue detected; original media retained.", {"video_id": video.id})
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
            current_audio_dur = ffmpeg_service.get_media_duration(
                dubbed_version.dubbed_audio_path
            )
            if abs(current_audio_dur - video.duration) > 1.0:
                audio_dur_mismatch = True

        need_tts = (
            need_translate
            or audio_dur_mismatch
            or not dubbed_version.dubbed_audio_path
            or not Path(dubbed_version.dubbed_audio_path).exists()
            or str(dubbed_audio_dest)
            != str(Path(dubbed_version.dubbed_audio_path).resolve())
            or (
                tts_engine.lower() == "xtts"
                and "_edge" in str(dubbed_version.dubbed_audio_path)
            )
        )

        if need_tts:
            emit("tts_prepare", 65.0, f"Configuring neural voice ({tts_engine}) for {target_language.upper()}...")
            logger.info(
                f"Synthesizing dubbed speech for video ID {video.id} in {target_language} (engine: {tts_engine})..."
            )
            # Extract clean speaker sample from original audio if voice cloning is active
            speaker_sample_path = None
            if video.original_audio_path:
                try:
                    start_offset = 0.0
                    if video.original_transcript_json and isinstance(
                        video.original_transcript_json, list
                    ):
                        for seg in video.original_transcript_json:
                            if isinstance(seg, dict) and seg.get("text", "").strip():
                                start_offset = float(seg.get("start", 0.0))
                                break
                    speaker_sample_path = ffmpeg_service.extract_speaker_sample(
                        video.original_audio_path,
                        start_time=start_offset,
                    )
                except Exception as e:
                    logger.warning(f"Could not extract speaker sample ({e}).")

            detected_gender = "male"
            ref_for_pitch = speaker_sample_path or video.original_audio_path
            if ref_for_pitch and Path(ref_for_pitch).exists():
                detected_gender = tts_service.estimate_gender(ref_for_pitch)

            segments_to_synth = self._extract_segments_from_transcript_json(
                dubbed_version.transcript_json
            )
            has_segments = len(segments_to_synth) > 0 and any(
                s.get("text", "").strip() for s in segments_to_synth
            )

            if has_segments:
                total_segs = len(segments_to_synth)
                logger.info(
                    f"Synthesizing {total_segs} segments on timeline for video ID {video.id}..."
                )
                clips_dir = (
                    ffmpeg_service.dubbed_audio_dir
                    / f"clips_{video.id}_{target_language}"
                )

                def on_tts_progress(done_idx: int, total_idx: int):
                    pct = 65.0 + (done_idx / max(total_idx, 1)) * 25.0
                    emit("tts", min(pct, 90.0), f"Synthesizing dubbed voice segment {done_idx}/{total_idx} ({tts_engine})...")

                clips = tts_service.synthesize_segments(
                    segments=segments_to_synth,
                    target_language=target_language,
                    output_dir=clips_dir,
                    speaker_wav_path=speaker_sample_path,
                    voice=voice,
                    engine=tts_engine,
                    gender=detected_gender,
                    on_progress=on_tts_progress,
                )
                total_media_dur = video.duration or ffmpeg_service.get_media_duration(
                    str(resolved_video)
                )
                dubbed_audio_path = ffmpeg_service.build_timeline_track(
                    clips=clips,
                    total_duration=total_media_dur,
                    output_path=str(dubbed_audio_dest),
                    sample_rate=24000,
                    max_speedup=1.35,
                )
            else:
                emit("tts", 75.0, f"Synthesizing full text directly via {tts_engine}...")
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
            emit("tts", 90.0, "Dubbed speech synthesis complete. Audio track finalized.")
        else:
            logger.info(
                f"Dubbed audio already present for video ID {video.id}: {dubbed_version.dubbed_audio_path}"
            )
            emit("tts", 90.0, "Dubbed audio stream already exists on disk.")

        need_merge = (
            not dubbed_version.dubbed_video_path
            or not Path(dubbed_version.dubbed_video_path).exists()
            or need_tts
        )

        if need_merge:
            emit("merge", 92.0, "Multiplexing dubbed audio with video stream (FFmpeg)...")
            logger.info(
                f"Merging dubbed audio into video for video ID {video.id}..."
            )
            dubbed_video_dest = (
                ffmpeg_service.dubbed_video_dir
                / f"{video.id}_{target_language}_{resolved_video.stem}.mp4"
            )
            dubbed_video_path = ffmpeg_service.merge_audio_video(
                video_path=str(resolved_video),
                audio_path=dubbed_version.dubbed_audio_path,
                output_path=str(dubbed_video_dest),
            )
            stored_dubbed_path = f"/data/videos/dubbed/{dubbed_video_dest.name}"
            dubbed_version.dubbed_video_path = stored_dubbed_path
            dubbed_version.status = "completed"
            db.commit()
            db.refresh(dubbed_version)
            emit("merge", 98.0, "Video multiplexing complete.")
        else:
            dubbed_version.status = "completed"
            db.commit()
            emit("merge", 98.0, "Dubbed video stream verified.")

        emit("completed", 100.0, f"Dubbed video in {target_language.upper()} generated successfully!", {"video_id": video.id})

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