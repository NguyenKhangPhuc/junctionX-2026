"""Text-to-Speech (TTS) service using edge-tts for high-quality multilingual neural voices."""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

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
}


class TTSService:
    """Service synthesizing speech audio using edge-tts neural voices."""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir).resolve()
        self.dubbed_audio_dir = self.data_dir / "audio" / "dubbed"
        self.dubbed_audio_dir.mkdir(parents=True, exist_ok=True)

    def get_voice_for_language(self, language: str) -> str:
        """Resolve a suitable neural voice name for a given language code."""
        lang_key = language.lower()[:2]
        return DEFAULT_VOICE_MAP.get(lang_key, "en-US-GuyNeural")

    async def synthesize_speech_async(
        self,
        text: str,
        target_language: str,
        output_path: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> str:
        """Asynchronously synthesize text into speech audio using edge-tts.

        Args:
            text: Spoken content to synthesize.
            target_language: Target language code (e.g. 'en', 'vi').
            output_path: Optional file destination path.
            voice: Optional specific voice identifier.

        Returns:
            str: Absolute path to the generated audio file.

        Raises:
            ValueError: If input text is empty.
            RuntimeError: If TTS generation fails.
        """
        import edge_tts

        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Cannot synthesize speech from empty text.")

        selected_voice = voice or self.get_voice_for_language(target_language)

        if output_path:
            out_file = Path(output_path).resolve()
        else:
            out_file = self.dubbed_audio_dir / f"tts_{target_language}_{abs(hash(clean_text)) % 1000000}.mp3"

        out_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Synthesizing speech with voice '{selected_voice}' for language '{target_language}' -> {out_file}"
        )
        try:
            communicate = edge_tts.Communicate(clean_text, selected_voice)
            await communicate.save(str(out_file))
            return str(out_file)
        except Exception as e:
            logger.error(f"edge-tts synthesis failed: {e}", exc_info=True)
            raise RuntimeError(f"Text-to-speech synthesis failed: {e}") from e

    def synthesize_speech(
        self,
        text: str,
        target_language: str,
        output_path: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> str:
        """Synchronous wrapper for synthesize_speech_async.

        Args:
            text: Text to synthesize.
            target_language: Target language code.
            output_path: Optional destination path.
            voice: Optional voice name.

        Returns:
            str: Absolute path to generated audio file.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(
                self.synthesize_speech_async(text, target_language, output_path, voice)
            )
        else:
            return asyncio.run(
                self.synthesize_speech_async(text, target_language, output_path, voice)
            )


tts_service = TTSService()
