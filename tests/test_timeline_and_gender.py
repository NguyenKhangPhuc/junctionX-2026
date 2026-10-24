"""Unit tests for timeline track assembly, tempo scaling, and pitch/gender estimation."""

import wave
from pathlib import Path

import numpy as np
import pytest

from app.services.ffmpeg_service import ffmpeg_service
from app.services.tts_service import tts_service


def test_compute_tempo_bounds():
    # Clip fits within slot -> no speedup (1.0)
    assert ffmpeg_service.compute_tempo(clip_duration=2.0, slot_duration=4.0) == 1.0

    # Clip slightly longer than slot -> scaled to fit
    tempo = ffmpeg_service.compute_tempo(clip_duration=3.0, slot_duration=2.5, max_speedup=1.4)
    assert pytest.approx(tempo, rel=1e-2) == 1.2

    # Clip much longer than slot -> capped at max_speedup
    capped = ffmpeg_service.compute_tempo(clip_duration=6.0, slot_duration=2.0, max_speedup=1.4)
    assert capped == 1.4


def test_estimate_f0_synthetic_tones():
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 120 Hz tone (typical adult male pitch ~85-180 Hz)
    male_tone = 0.5 * np.sin(2 * np.pi * 120.0 * t).astype(np.float32)
    f0_male = tts_service.estimate_f0(male_tone, sample_rate=sr)
    assert f0_male is not None
    assert 110.0 <= f0_male <= 130.0

    # 220 Hz tone (typical adult female pitch ~165-255 Hz)
    female_tone = 0.5 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    f0_female = tts_service.estimate_f0(female_tone, sample_rate=sr)
    assert f0_female is not None
    assert 210.0 <= f0_female <= 230.0


def test_build_timeline_track_exact_duration(tmp_path: Path):
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

    clips = [
        {"path": str(clip1_path), "start": 1.0, "end": 2.0, "text": "one"},
        {"path": str(clip2_path), "start": 3.5, "end": 4.5, "text": "two"},
    ]

    total_duration = 6.0
    out_wav = tmp_path / "assembled_track.wav"

    assembled = ffmpeg_service.build_timeline_track(
        clips=clips,
        total_duration=total_duration,
        output_path=str(out_wav),
        sample_rate=sr,
    )

    assert Path(assembled).exists()
    with wave.open(assembled, "rb") as wf:
        frames = wf.getnframes()
        actual_dur = frames / wf.getframerate()
        assert pytest.approx(actual_dur, abs=0.01) == total_duration
