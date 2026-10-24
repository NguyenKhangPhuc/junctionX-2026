"""Unit tests for timeline track assembly with silence padding and segment TTS synthesis."""

import wave
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from app.services.ffmpeg_service import ffmpeg_service
from app.services.tts_service import tts_service


def test_timeline_assembly_silence_padding(tmp_path: Path):
    sr = 24000
    # Create two synthetic mono WAV clips (1 second each)
    clip1_path = tmp_path / "clip1.wav"
    clip2_path = tmp_path / "clip2.wav"

    t = np.linspace(0, 1.0, sr, endpoint=False)
    pcm = (np.sin(2 * np.pi * 440.0 * t) * 16000).astype(np.int16).tobytes()

    for p in (clip1_path, clip2_path):
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm)

    # First speech starts at 4.18s, second speech starts at 8.0s
    clips = [
        {"path": str(clip1_path), "start": 4.18, "end": 5.18, "text": "first"},
        {"path": str(clip2_path), "start": 8.0, "end": 9.0, "text": "second"},
    ]

    total_duration = 12.0
    out_wav = tmp_path / "assembled_master.wav"

    assembled = ffmpeg_service.build_timeline_track(
        clips=clips,
        total_duration=total_duration,
        output_path=str(out_wav),
        sample_rate=sr,
    )

    assert Path(assembled).exists()

    data, sample_rate = sf.read(assembled, dtype="float32")
    actual_dur = len(data) / sample_rate
    assert pytest.approx(actual_dur, abs=0.01) == total_duration

    # 1. Exact silence from 0.0s to 4.10s (before speech begins at 4.18s)
    silence_start = data[: int(4.10 * sr)]
    assert np.max(np.abs(silence_start)) == 0.0

    # 2. Speech present from 4.18s to 5.18s
    speech_1 = data[int(4.18 * sr) : int(5.18 * sr)]
    assert np.max(np.abs(speech_1)) > 0.1

    # 3. Inter-segment silence from 5.30s to 7.90s
    silence_middle = data[int(5.30 * sr) : int(7.90 * sr)]
    assert np.max(np.abs(silence_middle)) == 0.0

    # 4. Speech present from 8.0s to 9.0s
    speech_2 = data[int(8.0 * sr) : int(9.0 * sr)]
    assert np.max(np.abs(speech_2)) > 0.1

    # 5. Trailing silence from 9.20s to 12.0s
    silence_end = data[int(9.20 * sr) :]
    assert np.max(np.abs(silence_end)) == 0.0


def test_synthesize_segments_edge(tmp_path: Path):
    segments = [
        {"start": 2.5, "end": 4.0, "text": "Hello world."},
        {"start": 5.5, "end": 7.0, "text": "This is a test."},
    ]
    clips_dir = tmp_path / "clips"
    clips = tts_service.synthesize_segments(
        segments=segments,
        target_language="en",
        output_dir=clips_dir,
        engine="edge",
    )

    assert len(clips) == 2
    for c in clips:
        assert Path(c["path"]).exists()
        assert c["start"] in (2.5, 5.5)

    out_wav = tmp_path / "assembled_edge.wav"
    assembled = ffmpeg_service.build_timeline_track(
        clips=clips,
        total_duration=10.0,
        output_path=str(out_wav),
        sample_rate=24000,
    )

    data, sr = sf.read(assembled, dtype="float32")
    assert pytest.approx(len(data) / sr, abs=0.05) == 10.0

    # Verify initial silence from 0.0s to 2.4s
    assert np.max(np.abs(data[: int(2.4 * sr)])) == 0.0

    # Verify speech starts around 2.5s
    assert np.max(np.abs(data[int(2.5 * sr) : int(3.5 * sr)])) > 0.05


def test_empty_segments_handling(tmp_path: Path):
    clips_dir = tmp_path / "empty_clips"
    assert tts_service.synthesize_segments([], "en", clips_dir) == []
    assert tts_service.synthesize_segments([{"start": 1.0, "end": 2.0, "text": "   "}], "en", clips_dir) == []


def test_overlapping_clips_sequential_placement(tmp_path: Path):
    sr = 24000
    # Two 2-second clips
    clip1_path = tmp_path / "clip1.wav"
    clip2_path = tmp_path / "clip2.wav"

    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    pcm = (np.sin(2 * np.pi * 440.0 * t) * 16000).astype(np.int16).tobytes()

    for p in (clip1_path, clip2_path):
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm)

    # Clip 1 starts at 1.0s and takes 2.0s (ends at 3.0s).
    # Clip 2 is scheduled at 2.0s (collides/overlaps with Clip 1 by 1.0s!).
    clips = [
        {"path": str(clip1_path), "start": 1.0, "end": 2.5, "text": "first"},
        {"path": str(clip2_path), "start": 2.0, "end": 3.5, "text": "second"},
    ]

    out_wav = tmp_path / "no_overlap.wav"
    assembled = ffmpeg_service.build_timeline_track(
        clips=clips,
        total_duration=6.0,
        output_path=str(out_wav),
        sample_rate=sr,
    )

    data, _ = sf.read(assembled, dtype="float32")
    # Clip 1 should start at 1.0s and finish at 3.0s
    # Clip 2 must be pushed to start after 3.0s (around 3.05s)
    # Silence at start: 0.0s to 0.95s
    assert np.max(np.abs(data[: int(0.95 * sr)])) == 0.0

    # Between 3.00s and 3.05s there is the micro-pause gap
    micro_pause = data[int(3.01 * sr) : int(3.04 * sr)]
    assert np.max(np.abs(micro_pause)) == 0.0

    # Clip 2 is playing around 3.10s - 4.50s
    assert np.max(np.abs(data[int(3.10 * sr) : int(4.50 * sr)])) > 0.1

