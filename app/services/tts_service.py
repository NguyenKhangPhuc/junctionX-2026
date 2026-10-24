"""Text-to-Speech (TTS) service with dual-engine architecture: Coqui XTTS-v2 and Edge-TTS."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

# Automatically accept Coqui non-commercial license without interactive stdin prompts
os.environ["COQUI_TOS_AGREED"] = "1"

logger = logging.getLogger(__name__)

# Languages supported officially by Coqui XTTS-v2
XTTS_SUPPORTED_LANGUAGES: Set[str] = {
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "pl",
    "tr",
    "ru",
    "nl",
    "cs",
    "ar",
    "ja",
    "hu",
    "ko",
    "hi",
}

LANGUAGE_ALIAS_MAP: Dict[str, str] = {
    "france": "fr",
    "french": "fr",
    "francais": "fr",
    "english": "en",
    "spain": "es",
    "spanish": "es",
    "germany": "de",
    "german": "de",
    "italy": "it",
    "italian": "it",
    "vietnam": "vi",
    "vietnamese": "vi",
    "japan": "ja",
    "japanese": "ja",
    "russia": "ru",
    "russian": "ru",
    "finland": "fi",
    "finnish": "fi",
}

DEFAULT_VOICE_MAP: Dict[str, Dict[str, str]] = {
    "vi": {"female": "vi-VN-HoaiMyNeural", "male": "vi-VN-NamMinhNeural"},
    "en": {"female": "en-US-JennyNeural", "male": "en-US-GuyNeural"},
    "fr": {"female": "fr-FR-DeniseNeural", "male": "fr-FR-HenriNeural"},
    "es": {"female": "es-ES-ElviraNeural", "male": "es-ES-AlvaroNeural"},
    "de": {"female": "de-DE-KatjaNeural", "male": "de-DE-KillianNeural"},
    "ja": {"female": "ja-JP-NanamiNeural", "male": "ja-JP-KeitaNeural"},
    "zh": {"female": "zh-CN-XiaoxiaoNeural", "male": "zh-CN-YunxiNeural"},
    "ko": {"female": "ko-KR-SunHiNeural", "male": "ko-KR-InJoonNeural"},
    "ru": {"female": "ru-RU-SvetlanaNeural", "male": "ru-RU-DmitryNeural"},
    "fi": {"female": "fi-FI-NooraNeural", "male": "fi-FI-HarriNeural"},
}


class TTSService:
    """Service providing voice cloning via XTTS-v2 and fast broadcast TTS via Edge-TTS."""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir).resolve()
        self.dubbed_audio_dir = self.data_dir / "audio" / "dubbed"
        self.dubbed_audio_dir.mkdir(parents=True, exist_ok=True)
        self._xtts_model = None

    def _patch_xtts_audio_loader(self):
        """Monkeypatch XTTS load_audio with soundfile to avoid torchcodec/libnvrtc CUDA crashes on CPU."""
        try:
            import soundfile as sf
            import torch
            import torchaudio.functional as F
            import TTS.tts.models.xtts as xtts_mod

            def custom_load_audio(audiopath, sampling_rate):
                data, sr = sf.read(str(audiopath), dtype="float32")
                if data.ndim == 1:
                    audio = torch.from_numpy(data).unsqueeze(0)
                else:
                    audio = torch.from_numpy(data.T).mean(dim=0, keepdim=True)
                if sr != sampling_rate:
                    audio = F.resample(audio, sr, sampling_rate)
                return audio

            xtts_mod.load_audio = custom_load_audio
            logger.info("Successfully patched XTTS load_audio with soundfile backend.")
        except Exception as e:
            logger.warning(f"Could not patch XTTS load_audio: {e}")

    def _get_xtts_model(self):
        """Lazily initialize the Coqui XTTS-v2 model."""
        if self._xtts_model is None:
            self._patch_xtts_audio_loader()
            import torch
            from TTS.api import TTS

            use_gpu = torch.cuda.is_available()
            device_str = "cuda" if use_gpu else "cpu"
            logger.info(f"Loading Coqui XTTS-v2 model on {device_str}...")
            self._xtts_model = TTS(
                "tts_models/multilingual/multi-dataset/xtts_v2",
                gpu=use_gpu,
            )
            logger.info("Coqui XTTS-v2 model initialized successfully.")
        return self._xtts_model

    @staticmethod
    def estimate_f0(audio: Any, sample_rate: int = 16000) -> Optional[float]:
        """Estimate fundamental frequency (pitch) of an audio signal using autocorrelation.

        Args:
            audio: numpy array of samples or path to audio file.
            sample_rate: sampling rate of the signal.

        Returns:
            Estimated pitch in Hz, or None if signal is silent/too short.
        """
        import numpy as np

        if isinstance(audio, (str, Path)):
            import soundfile as sf

            sig, sr = sf.read(str(audio), dtype="float32")
        else:
            sig = np.asarray(audio, dtype=np.float32)
            sr = sample_rate

        if sig.ndim > 1:
            sig = sig.mean(axis=1 if sig.shape[1] < sig.shape[0] else 0)

        if len(sig) < int(sr * 0.05):
            return None

        # Center signal
        sig = sig - float(np.mean(sig))

        n = len(sig)
        f_sig = np.fft.rfft(sig, n=2 * n)
        autocorr = np.fft.irfft(f_sig * np.conj(f_sig))[:n]

        min_lag = max(1, int(sr / 350.0))  # Max 350 Hz
        max_lag = min(n - 1, int(sr / 70.0))  # Min 70 Hz

        if max_lag <= min_lag:
            return None

        peak_offset = int(np.argmax(autocorr[min_lag:max_lag]))
        peak_lag = min_lag + peak_offset
        if peak_lag == 0:
            return None

        return float(sr / peak_lag)

    def estimate_gender(self, audio: Any, sample_rate: int = 16000) -> str:
        """Estimate speaker gender ('female' or 'male') from pitch.

        Typical adult female f0: ~165-260 Hz, male f0: ~85-165 Hz.
        """
        f0 = self.estimate_f0(audio, sample_rate)
        if f0 is not None and f0 >= 165.0:
            return "female"
        return "male"

    @staticmethod
    def normalize_language(language: str) -> str:
        """Map language name or alias to standardized ISO code."""
        cleaned = language.lower().strip()
        if cleaned in LANGUAGE_ALIAS_MAP:
            return LANGUAGE_ALIAS_MAP[cleaned]
        if len(cleaned) > 2 and cleaned[:2] in XTTS_SUPPORTED_LANGUAGES:
            return cleaned[:2]
        return cleaned

    def get_voice_for_language(self, language: str, gender: str = "male") -> str:
        """Resolve a suitable neural voice name for a given language code and gender for Edge-TTS."""
        lang_key = self.normalize_language(language)
        if len(lang_key) > 2 and lang_key not in {"zh-cn"}:
            lang_key = lang_key[:2]
        gender_map = DEFAULT_VOICE_MAP.get(lang_key, DEFAULT_VOICE_MAP["en"])
        resolved_gender = gender.lower() if gender else "male"
        return gender_map.get(resolved_gender, gender_map.get("male", "en-US-GuyNeural"))

    async def _synthesize_edge_async(
        self,
        text: str,
        target_language: str,
        output_path: Optional[str] = None,
        voice: Optional[str] = None,
        gender: str = "male",
    ) -> str:
        """Asynchronously synthesize text using Microsoft Edge-TTS."""
        import edge_tts

        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Cannot synthesize speech from empty text.")

        selected_voice = voice or self.get_voice_for_language(target_language, gender=gender)

        if output_path:
            out_file = Path(output_path).resolve()
        else:
            out_file = (
                self.dubbed_audio_dir
                / f"tts_edge_{target_language}_{abs(hash(clean_text)) % 1000000}.mp3"
            )

        out_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[Edge-TTS] Synthesizing speech with voice '{selected_voice}' for '{target_language}' -> {out_file}"
        )
        try:
            communicate = edge_tts.Communicate(clean_text, selected_voice)
            await communicate.save(str(out_file))
            return str(out_file)
        except Exception as e:
            logger.error(f"Edge-TTS synthesis failed: {e}", exc_info=True)
            raise RuntimeError(f"Edge-TTS synthesis failed: {e}") from e

    def _synthesize_edge_sync(
        self,
        text: str,
        target_language: str,
        output_path: Optional[str] = None,
        voice: Optional[str] = None,
        gender: str = "male",
    ) -> str:
        """Synchronous wrapper for Edge-TTS."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self._synthesize_edge_async(text, target_language, output_path, voice, gender=gender),
                )
                return future.result()
        else:
            return asyncio.run(
                self._synthesize_edge_async(text, target_language, output_path, voice, gender=gender)
            )

    def synthesize_speech(
        self,
        text: str,
        target_language: str,
        speaker_wav_path: Optional[str] = None,
        output_path: Optional[str] = None,
        voice: Optional[str] = None,
        engine: str = "xtts",
        gender: Optional[str] = None,
    ) -> str:
        """Synthesize speech using either Coqui XTTS-v2 voice cloning or Edge-TTS.

        Args:
            text: Text script to synthesize.
            target_language: Target language code (e.g. 'en', 'vi').
            speaker_wav_path: Path to reference speaker audio file (required for XTTS-v2 voice cloning).
            output_path: Optional destination audio file path.
            voice: Optional specific Edge-TTS voice name override.
            engine: Engine to use: 'xtts' for voice cloning, 'edge' for broadcast neural voice.
            gender: Speaker gender ('female' or 'male') for voice selection in Edge-TTS.

        Returns:
            str: Path to generated audio file.
        """
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Cannot synthesize speech from empty text.")

        normalized_lang = self.normalize_language(target_language)
        resolved_gender = gender
        if not resolved_gender and speaker_wav_path and Path(speaker_wav_path).exists():
            resolved_gender = self.estimate_gender(speaker_wav_path)
        if not resolved_gender:
            resolved_gender = "male"

        # Check if user requested XTTS-v2 voice cloning
        if engine.lower() == "xtts":
            # 1. Check language support
            if normalized_lang not in XTTS_SUPPORTED_LANGUAGES:
                logger.warning(
                    f"XTTS-v2 does not support target language '{target_language}'. "
                    f"Auto-falling back to Edge-TTS with neural voice ({resolved_gender})."
                )
                return self._synthesize_edge_sync(
                    clean_text, target_language, output_path, voice, gender=resolved_gender
                )

            # 2. Check speaker wav reference audio
            if not speaker_wav_path or not Path(speaker_wav_path).exists():
                logger.warning(
                    f"Reference audio '{speaker_wav_path}' not found for XTTS-v2 voice cloning. "
                    f"Auto-falling back to Edge-TTS ({resolved_gender})."
                )
                return self._synthesize_edge_sync(
                    clean_text, target_language, output_path, voice, gender=resolved_gender
                )

            # 3. Execute XTTS-v2 Voice Cloning
            if output_path:
                out_file = Path(output_path).resolve()
            else:
                out_file = (
                    self.dubbed_audio_dir
                    / f"tts_xtts_{normalized_lang}_{abs(hash(clean_text)) % 1000000}.wav"
                )

            out_file.parent.mkdir(parents=True, exist_ok=True)

            logger.info(
                f"[XTTS-v2] Cloning voice from '{speaker_wav_path}' speaking '{normalized_lang}' -> {out_file}"
            )
            try:
                model = self._get_xtts_model()
                model.tts_to_file(
                    text=clean_text,
                    speaker_wav=str(speaker_wav_path),
                    language=normalized_lang,
                    file_path=str(out_file),
                )
                return str(out_file)
            except Exception as e:
                logger.error(f"[XTTS-v2] Synthesis failed: {e}", exc_info=True)
                raise RuntimeError(f"XTTS-v2 synthesis failed: {e}") from e

        # Default / Explicit Edge-TTS
        return self._synthesize_edge_sync(clean_text, target_language, output_path, voice, gender=resolved_gender)

    async def _synthesize_edge_segments_async(
        self,
        segments: List[Dict[str, Any]],
        target_language: str,
        output_dir: Union[str, Path],
        voice: Optional[str] = None,
        gender: str = "male",
    ) -> List[Dict[str, Any]]:
        """Asynchronously synthesize multiple speech segments concurrently using Edge-TTS."""
        import edge_tts

        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        selected_voice = voice or self.get_voice_for_language(target_language, gender=gender)

        async def _synth_one(idx: int, seg: Dict[str, Any]):
            text = seg.get("text", "").strip()
            if not text:
                return None
            clip_file = out_path / f"clip_{idx}.mp3"
            communicate = edge_tts.Communicate(text, selected_voice)
            await communicate.save(str(clip_file))
            return {
                "path": str(clip_file),
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": text,
            }

        tasks = [_synth_one(i, s) for i, s in enumerate(segments)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        clips = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                logger.error(f"Failed to synthesize segment {i}: {res}")
            elif res is not None:
                clips.append(res)
        return clips

    def synthesize_segments(
        self,
        segments: List[Dict[str, Any]],
        target_language: str,
        output_dir: Union[str, Path],
        speaker_wav_path: Optional[str] = None,
        voice: Optional[str] = None,
        engine: str = "xtts",
        gender: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Synthesize a sequence of time-coded segments into individual audio clip files.

        Args:
            segments: List of dicts with {"start": float, "end": float, "text": str}.
            target_language: Target language code.
            output_dir: Directory where segment clips will be stored.
            speaker_wav_path: Reference speaker wav for voice cloning (XTTS).
            voice: Voice override for Edge-TTS.
            engine: 'xtts' or 'edge'.
            gender: Speaker gender ('female' or 'male').

        Returns:
            List[Dict[str, Any]]: List of clips with 'path', 'start', 'end', and 'text'.
        """
        if not segments:
            return []

        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        normalized_lang = self.normalize_language(target_language)
        resolved_gender = gender
        if not resolved_gender and speaker_wav_path and Path(speaker_wav_path).exists():
            resolved_gender = self.estimate_gender(speaker_wav_path)
        if not resolved_gender:
            resolved_gender = "male"

        use_xtts = (
            engine.lower() == "xtts"
            and normalized_lang in XTTS_SUPPORTED_LANGUAGES
            and speaker_wav_path
            and Path(speaker_wav_path).exists()
        )

        if use_xtts:
            clips = []
            try:
                model = self._get_xtts_model()
                for idx, seg in enumerate(segments):
                    text = seg.get("text", "").strip()
                    if not text:
                        continue
                    clip_file = out_path / f"clip_{idx}.wav"
                    model.tts_to_file(
                        text=text,
                        speaker_wav=str(speaker_wav_path),
                        language=normalized_lang,
                        file_path=str(clip_file),
                    )
                    clips.append(
                        {
                            "path": str(clip_file),
                            "start": float(seg.get("start", 0.0)),
                            "end": float(seg.get("end", 0.0)),
                            "text": text,
                        }
                    )
                return clips
            except Exception as e:
                logger.warning(
                    f"XTTS segment synthesis failed ({e}). Falling back to Edge-TTS."
                )

        # Edge-TTS (either chosen directly or fallback)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self._synthesize_edge_segments_async(
                        segments, target_language, out_path, voice=voice, gender=resolved_gender
                    ),
                )
                return future.result()
        else:
            return asyncio.run(
                self._synthesize_edge_segments_async(
                    segments, target_language, out_path, voice=voice, gender=resolved_gender
                )
            )


tts_service = TTSService()
