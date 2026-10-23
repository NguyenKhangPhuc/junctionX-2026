"""FFmpeg utility service for audio extraction, duration probing, and audio-video merging."""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FFmpegService:
    """Service wrapping OS-level FFmpeg and FFprobe CLI operations."""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir).resolve()
        self.raw_audio_dir = self.data_dir / "audio" / "raw"
        self.dubbed_audio_dir = self.data_dir / "audio" / "dubbed"
        self.dubbed_video_dir = self.data_dir / "videos" / "dubbed"

        # Ensure directories exist
        self.raw_audio_dir.mkdir(parents=True, exist_ok=True)
        self.dubbed_audio_dir.mkdir(parents=True, exist_ok=True)
        self.dubbed_video_dir.mkdir(parents=True, exist_ok=True)

    def extract_audio(
        self, video_path: str, output_path: Optional[str] = None
    ) -> str:
        """Extract a 16kHz mono WAV audio file from a source video for optimal STT accuracy.

        Args:
            video_path: Path to the source video file.
            output_path: Optional destination path for the extracted audio.

        Returns:
            str: Absolute path to the extracted WAV audio file.

        Raises:
            FileNotFoundError: If the source video file does not exist.
            RuntimeError: If FFmpeg execution fails.
        """
        video_file = Path(video_path).resolve()
        if not video_file.exists():
            raise FileNotFoundError(f"Source video file not found at: {video_path}")

        if output_path:
            out_file = Path(output_path).resolve()
        else:
            out_file = self.raw_audio_dir / f"{video_file.stem}.wav"

        out_file.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_file),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(out_file),
        ]

        logger.info(f"Extracting audio using command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            logger.error(f"FFmpeg audio extraction failed: {result.stderr}")
            raise RuntimeError(
                f"FFmpeg audio extraction failed with exit code {result.returncode}: {result.stderr}"
            )

        return str(out_file)

    def extract_speaker_sample(
        self, audio_path: str, output_path: Optional[str] = None, duration: float = 6.0
    ) -> str:
        """Extract a clean 3-10s WAV audio segment (22050Hz mono) for voice cloning reference.

        Args:
            audio_path: Path to original audio file.
            output_path: Optional destination path for speaker sample.
            duration: Sample length in seconds (default: 6.0s).

        Returns:
            str: Path to the extracted reference speaker WAV file.

        Raises:
            FileNotFoundError: If audio file does not exist.
            RuntimeError: If FFmpeg extraction fails.
        """
        a_file = Path(audio_path).resolve()
        if not a_file.exists():
            raise FileNotFoundError(f"Audio file for speaker sample not found: {audio_path}")

        if output_path:
            out_file = Path(output_path).resolve()
        else:
            out_file = self.raw_audio_dir / f"{a_file.stem}_speaker_sample.wav"

        out_file.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(a_file),
            "-ss",
            "0",
            "-t",
            str(duration),
            "-acodec",
            "pcm_s16le",
            "-ar",
            "22050",
            "-ac",
            "1",
            str(out_file),
        ]

        logger.info(f"Extracting speaker reference sample: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            logger.error(f"Failed to extract speaker sample: {result.stderr}")
            raise RuntimeError(
                f"Failed to extract speaker sample with exit code {result.returncode}: {result.stderr}"
            )

        return str(out_file)

    def get_media_duration(self, file_path: str) -> float:
        """Probe the duration of an audio or video file using ffprobe.

        Args:
            file_path: Path to the target media file.

        Returns:
            float: Duration in seconds (0.0 if cannot be determined).
        """
        target = Path(file_path).resolve()
        if not target.exists():
            return 0.0

        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(target),
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            val = result.stdout.strip()
            return float(val) if val else 0.0
        except Exception as e:
            logger.warning(f"Could not determine duration for {file_path}: {e}")
            return 0.0

    def merge_audio_video(
        self, video_path: str, audio_path: str, output_path: Optional[str] = None
    ) -> str:
        """Combine video stream from source video with a new audio track.

        Replaces the audio track in the video and truncates to the shortest stream.

        Args:
            video_path: Source video file path.
            audio_path: New audio track file path.
            output_path: Optional path for the resulting video.

        Returns:
            str: Absolute path to the merged dubbed video file.

        Raises:
            FileNotFoundError: If either video or audio file does not exist.
            RuntimeError: If FFmpeg execution fails.
        """
        v_file = Path(video_path).resolve()
        a_file = Path(audio_path).resolve()

        if not v_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not a_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if output_path:
            out_file = Path(output_path).resolve()
        else:
            out_file = self.dubbed_video_dir / f"{v_file.stem}_dubbed.mp4"

        out_file.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(v_file),
            "-i",
            str(a_file),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            str(out_file),
        ]

        logger.info(f"Merging audio and video using command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            logger.error(f"FFmpeg merge failed: {result.stderr}")
            raise RuntimeError(
                f"FFmpeg merge failed with exit code {result.returncode}: {result.stderr}"
            )

        return str(out_file)


ffmpeg_service = FFmpegService()
