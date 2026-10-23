"""Text-to-Speech (TTS) service with dual-engine architecture: Coqui XTTS-v2 and Edge-TTS."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Set

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
    "zh-cn",
    "ja",
    "hu",
    "ko",
    "hi",
}

DEFAULT_VOICE_MAP: Dict[str, str] = {
    "vi": "vi-VN-HoaiMyNeural",
    "en": "en-US-GuyNeural",
    "fr": "fr-FR-HenriNeural",
    "es": "es-ES-AlvaroNeural",
    "de": "de-DE-KillianNeural",
    "ja": "ja-JP-KeitaNeural",
    "zh": "zh-CN-YunxiNeural",
    "ko": "ko-KR-InJoonNeural",
    "ru": "ru-RU-DmitryNeural",
    "fi": "fi-FI-HarriNeural",
}


class TTSService:
    """Service providing voice cloning via XTTS-v2 and fast broadcast TTS via Edge-TTS."""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir).resolve()
        self.dubbed_audio_dir = self.data_dir / "audio" / "dubbed"
        self.dubbed_audio_dir.mkdir(parents=True, exist_ok=True)
        self._xtts_model = None

    def _get_xtts_model(self):
        """Lazily initialize the Coqui XTTS-v2 model."""
        if self._xtts_model is None:
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

    def get_voice_for_language(self, language: str) -> str:
        """Resolve a suitable neural voice name for a given language code for Edge-TTS."""
        lang_key = language.lower()[:2]
        return DEFAULT_VOICE_MAP.get(lang_key, "en-US-GuyNeural")

    async def _synthesize_edge_async(
        self,
        text: str,
        target_language: str,
        output_path: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> str:
        """Asynchronously synthesize text using Microsoft Edge-TTS."""
        import edge_tts

        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Cannot synthesize speech from empty text.")

        selected_voice = voice or self.get_voice_for_language(target_language)

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
                    self._synthesize_edge_async(text, target_language, output_path, voice),
                )
                return future.result()
        else:
            return asyncio.run(
                self._synthesize_edge_async(text, target_language, output_path, voice)
            )

    def synthesize_speech(
        self,
        text: str,
        target_language: str,
        speaker_wav_path: Optional[str] = None,
        output_path: Optional[str] = None,
        voice: Optional[str] = None,
        engine: str = "xtts",
    ) -> str:
        """Synthesize speech using either Coqui XTTS-v2 voice cloning or Edge-TTS.

        Args:
            text: Text script to synthesize.
            target_language: Target language code (e.g. 'en', 'vi').
            speaker_wav_path: Path to reference speaker audio file (required for XTTS-v2 voice cloning).
            output_path: Optional destination audio file path.
            voice: Optional specific Edge-TTS voice name override.
            engine: Engine to use: 'xtts' for voice cloning, 'edge' for broadcast neural voice.

        Returns:
            str: Path to generated audio file.
        """
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Cannot synthesize speech from empty text.")

        normalized_lang = target_language.lower().strip()
        if normalized_lang == "zh":
            normalized_lang = "zh-cn"

        # Check if user requested XTTS-v2 voice cloning
        if engine.lower() == "xtts":
            # 1. Check language support
            if normalized_lang not in XTTS_SUPPORTED_LANGUAGES:
                logger.warning(
                    f"XTTS-v2 does not support target language '{target_language}'. "
                    f"Auto-falling back to Edge-TTS with neural voice."
                )
                return self._synthesize_edge_sync(
                    clean_text, target_language, output_path, voice
                )

            # 2. Check speaker wav reference audio
            if not speaker_wav_path or not Path(speaker_wav_path).exists():
                logger.warning(
                    f"Reference audio '{speaker_wav_path}' not found for XTTS-v2 voice cloning. "
                    f"Auto-falling back to Edge-TTS."
                )
                return self._synthesize_edge_sync(
                    clean_text, target_language, output_path, voice
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
                logger.error(
                    f"[XTTS-v2] Synthesis failed ({e}). Falling back to Edge-TTS...",
                    exc_info=True,
                )
                return self._synthesize_edge_sync(
                    clean_text, target_language, output_path, voice
                )

        # Default / Explicit Edge-TTS
        return self._synthesize_edge_sync(clean_text, target_language, output_path, voice)


tts_service = TTSService()
