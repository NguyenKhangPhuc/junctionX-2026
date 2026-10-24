"""Speech-to-Text (STT) service using faster-whisper with timestamp extraction."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class STTService:
    """Service wrapping faster-whisper for accurate, time-coded speech transcription."""

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _get_model(self):
        """Lazily initialize the WhisperModel to conserve resources until needed."""
        if self._model is None:
            from faster_whisper import WhisperModel

            logger.info(
                f"Loading faster-whisper model '{self.model_size}' on {self.device} ({self.compute_type})..."
            )
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("faster-whisper model loaded successfully.")
        return self._model

    def transcribe_audio(
        self, audio_path: str, language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Transcribe an audio file into text and time-coded segments.

        Args:
            audio_path: Path to the target audio file (.wav, .mp3, etc.).
            language: Optional ISO language code (e.g., 'vi', 'en'). If omitted, language is detected.

        Returns:
            Dict containing:
                - language: Detected or provided language code
                - language_probability: Confidence score for language detection
                - full_text: Combined string of all transcription segments
                - segments: List of segment dicts with start, end, and text

        Raises:
            FileNotFoundError: If audio file does not exist.
            RuntimeError: If transcription encounters an unrecoverable failure.
        """
        target_path = Path(audio_path).resolve()
        if not target_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model = self._get_model()

        logger.info(f"Starting speech transcription for: {target_path}")
        try:
            segments_generator, info = model.transcribe(
                str(target_path),
                language=language,
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=400,
                    speech_pad_ms=200,
                ),
            )

            segments: List[Dict[str, Any]] = []
            full_text_parts: List[str] = []

            for seg in segments_generator:
                text_clean = seg.text.strip()
                if not text_clean:
                    continue
                segments.append(
                    {
                        "start": round(seg.start, 2),
                        "end": round(seg.end, 2),
                        "text": text_clean,
                    }
                )
                full_text_parts.append(text_clean)

            full_text = " ".join(full_text_parts)
            detected_lang = info.language if info and hasattr(info, "language") else (language or "unknown")
            lang_prob = info.language_probability if info and hasattr(info, "language_probability") else 1.0

            logger.info(
                f"Transcription complete. Language: {detected_lang} (prob: {lang_prob:.2f}), Segments: {len(segments)}"
            )

            return {
                "language": detected_lang,
                "language_probability": round(lang_prob, 4),
                "full_text": full_text,
                "segments": segments,
            }

        except Exception as e:
            logger.error(f"Error during audio transcription: {e}", exc_info=True)
            raise RuntimeError(f"Audio transcription failed: {e}") from e


stt_service = STTService()
