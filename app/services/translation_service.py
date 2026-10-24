"""Translation service supporting OpenRouter and local Ollama fallback."""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class TranslationService:
    """Service translating text and structured time-coded segments using LLMs."""

    def __init__(self):
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.ollama_model = os.getenv("LLM_MODEL", "llama3.2")

    @property
    def api_key(self) -> Optional[str]:
        """Dynamic lookup for OpenRouter API key."""
        return os.getenv("OPENROUTER_API_KEY") or self.openrouter_api_key

    @property
    def model_name(self) -> str:
        """Dynamic lookup for OpenRouter Model."""
        return os.getenv("OPENROUTER_MODEL") or self.openrouter_model

    def _call_openrouter(self, system_prompt: str, user_prompt: str) -> Tuple[Optional[str], Optional[str]]:
        """Call OpenRouter API using OpenAI SDK format via httpx.

        Returns:
            Tuple of (response_text, error_message).
        """
        api_key = self.api_key
        if not api_key:
            return None, "OPENROUTER_API_KEY not configured"

        model = self.model_name
        models_list = [
            model,
            "google/gemini-2.5-flash-lite",
            "meta-llama/llama-3.1-8b-instruct",
        ]
        # OpenRouter strictly enforces a maximum of 3 items in the 'models' array
        unique_models = list(dict.fromkeys(models_list))[:3]

        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:5605",
                "X-Title": "Video Dubbing Pipeline",
            }
            payload = {
                "models": unique_models,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 2048,
                "temperature": 0.2,
            }
            logger.info(f"Calling OpenRouter with models fallback {unique_models} for translation...")
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=45.0,
            )
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip(), None
            else:
                err_msg = f"HTTP {response.status_code}: {response.text}"
                logger.warning(f"OpenRouter returned status {err_msg}")
                return None, err_msg
        except Exception as e:
            logger.warning(f"Failed to query OpenRouter: {e}")
            return None, str(e)

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Fallback to local Ollama instance."""
        try:
            logger.info(f"Calling local Ollama model '{self.ollama_model}' at {self.ollama_host}...")
            payload = {
                "model": self.ollama_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "keep_alive": 0,
                "options": {"temperature": 0.3},
            }
            response = httpx.post(
                f"{self.ollama_host}/api/chat",
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Local Ollama call failed: {e}", exc_info=True)
            raise RuntimeError(f"Ollama error: {e}") from e

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Route to OpenRouter if configured, otherwise fallback to local Ollama."""
        openrouter_error = None
        if self.api_key:
            res, openrouter_error = self._call_openrouter(system_prompt, user_prompt)
            if res:
                return res

        try:
            return self._call_ollama(system_prompt, user_prompt)
        except Exception as ollama_err:
            if openrouter_error:
                raise RuntimeError(
                    f"All LLM translation backends failed. OpenRouter error: {openrouter_error} | Ollama error: {ollama_err}"
                ) from ollama_err
            raise RuntimeError(f"All LLM translation backends failed. {ollama_err}") from ollama_err

    def translate_segments(
        self,
        segments: List[Dict[str, Any]],
        source_lang: str,
        target_lang: str,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Translate a sequence of time-coded speech segments to target language.

        Preserves start/end timestamps and returns both the translated continuous text
        and updated segment list.

        Args:
            segments: List of segment dicts with {"start": float, "end": float, "text": str}.
            source_lang: Source language code (e.g., 'vi', 'en').
            target_lang: Target language code (e.g., 'en', 'vi').

        Returns:
            Tuple[str, List[Dict[str, Any]]]: (full_translated_text, translated_segments).
        """
        if not segments:
            return "", []

        # If source and target language are the same, return as-is
        if source_lang.lower() == target_lang.lower():
            full = " ".join([s["text"] for s in segments])
            return full, segments

        system_prompt = (
            f"You are a professional multilingual translator specializing in subtitle and news translation. "
            f"Translate the provided dialogue segments from {source_lang} to {target_lang}. "
            f"CRITICAL RULES:\n"
            f"1. You MUST maintain natural tone and matching meaning.\n"
            f"2. You MUST keep the exact same start and end timestamps for each item.\n"
            f"3. Output MUST be valid JSON array of objects with keys 'start', 'end', and 'text'.\n"
            f"4. Do NOT include markdown code fences (like ```json), explanations, or notes. ONLY output raw JSON."
        )

        user_prompt = json.dumps(segments, ensure_ascii=False)

        try:
            raw_response = self._call_llm(system_prompt, user_prompt)

            # Strip possible markdown code blocks if the LLM wrapped it
            cleaned = raw_response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            if isinstance(parsed, list) and len(parsed) > 0:
                translated_segments = []
                translated_texts = []
                for idx, orig in enumerate(segments):
                    # Ensure matching segment timestamps
                    if idx < len(parsed):
                        text = str(parsed[idx].get("text", orig["text"])).strip()
                    else:
                        text = orig["text"]

                    translated_segments.append(
                        {
                            "start": orig["start"],
                            "end": orig["end"],
                            "text": text,
                        }
                    )
                    translated_texts.append(text)

                full_translated_text = " ".join(translated_texts)
                return full_translated_text, translated_segments

        except Exception as e:
            logger.warning(
                f"Structured JSON translation failed ({e}). Falling back to whole text translation..."
            )

        # Fallback: Translate full text
        full_source = " ".join([s["text"] for s in segments])
        fb_sys = f"Translate the following text from {source_lang} to {target_lang}. Output only the translation."
        full_translated_text = self._call_llm(fb_sys, full_source).strip()

        # Update text across segments proportionally or use full text
        translated_segments = [
            {"start": s["start"], "end": s["end"], "text": s["text"]} for s in segments
        ]
        if translated_segments:
            translated_segments[0]["text"] = full_translated_text

        return full_translated_text, translated_segments


translation_service = TranslationService()
