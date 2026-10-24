"""FFmpeg utility service for audio extraction, duration probing, and audio-video merging."""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        self,
        audio_path: str,
        output_path: Optional[str] = None,
        start_time: float = 0.0,
        duration: float = 10.0,
    ) -> str:
        """Extract a clean 3-10s WAV audio segment (22050Hz mono) for voice cloning reference.

        Args:
            audio_path: Path to original audio file.
            output_path: Optional destination path for speaker sample.
            start_time: Offset in seconds where clear speech begins (default: 0.0s).
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
            "-ss",
            str(max(0.0, start_time)),
            "-i",
            str(a_file),
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

    @staticmethod
    def compute_tempo(
        clip_duration: float, slot_duration: float, max_speedup: float = 1.35
    ) -> float:
        """Compute the tempo speedup factor needed to fit clip_duration into slot_duration.

        Args:
            clip_duration: Duration of the synthesized audio in seconds.
            slot_duration: Target duration of the speech slot in seconds.
            max_speedup: Maximum allowed tempo scaling factor.

        Returns:
            float: Tempo scaling factor (1.0 to max_speedup).
        """
        if slot_duration <= 0.0 or clip_duration <= slot_duration:
            return 1.0
        return min(max_speedup, float(clip_duration / slot_duration))

    def adjust_clip_tempo(
        self, clip_path: str, tempo: float, output_path: Optional[str] = None
    ) -> str:
        """Apply atempo filter to an audio clip using FFmpeg."""
        if abs(tempo - 1.0) < 0.02:
            return clip_path

        c_file = Path(clip_path).resolve()
        if output_path:
            out_file = Path(output_path).resolve()
        else:
            out_file = c_file.parent / f"{c_file.stem}_tempo_{tempo:.2f}{c_file.suffix}"

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(c_file),
            "-filter:a",
            f"atempo={tempo:.3f}",
            str(out_file),
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                f"Failed to adjust clip tempo: {result.stderr}. Using original clip."
            )
            return clip_path
        return str(out_file)

    def build_timeline_track(
        self,
        clips: List[Dict[str, Any]],
        total_duration: float,
        output_path: str,
        sample_rate: int = 24000,
        max_speedup: float = 1.35,
    ) -> str:
        """Assemble time-coded audio clips onto a master timeline with exact silence padding.

        Args:
            clips: List of dicts with {"path": str, "start": float, "end": float, "text": str}.
            total_duration: Total duration of the target media in seconds.
            output_path: Destination file path for assembled master track.
            sample_rate: Output audio sampling rate in Hz (default: 24000).
            max_speedup: Maximum allowable speedup factor.

        Returns:
            str: Path to the generated master WAV audio track.
        """
        import numpy as np
        import soundfile as sf

        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        total_samples = max(1, int(round(total_duration * sample_rate)))
        master = np.zeros(total_samples, dtype=np.float32)

        for clip in clips:
            clip_path = clip.get("path")
            if not clip_path or not Path(clip_path).exists():
                continue

            start_sec = max(0.0, float(clip.get("start", 0.0)))
            end_sec = float(clip.get("end", start_sec))
            slot_dur = max(0.1, end_sec - start_sec)

            clip_dur = self.get_media_duration(clip_path)
            actual_path = clip_path
            tempo = self.compute_tempo(clip_dur, slot_dur, max_speedup=max_speedup)
            if tempo > 1.02:
                actual_path = self.adjust_clip_tempo(clip_path, tempo)

            data, sr = sf.read(actual_path, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)

            if sr != sample_rate:
                import torch
                import torchaudio.functional as F

                t_data = torch.from_numpy(data).unsqueeze(0)
                t_resampled = F.resample(t_data, sr, sample_rate)
                data = t_resampled.squeeze(0).numpy()

            start_idx = int(round(start_sec * sample_rate))
            end_idx = start_idx + len(data)

            if start_idx >= total_samples:
                continue

            if end_idx > total_samples:
                master[start_idx:] += data[: total_samples - start_idx]
            else:
                master[start_idx:end_idx] += data

        peak = float(np.max(np.abs(master)))
        if peak > 1.0:
            master = master / peak

        sf.write(str(out_file), master, sample_rate, subtype="PCM_16")
        return str(out_file)


ffmpeg_service = FFmpegService()
