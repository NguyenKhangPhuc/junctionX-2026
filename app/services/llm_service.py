"""LLM access layer for OpenRouter (hosted) and Ollama (local).

Supports streaming token deltas, non-streaming generation, and JSON extraction with retries —
the agent graph needs structured JSON from local models that often wrap output in prose or
code fences.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Sequence

import httpx

from app.config.settings import settings

logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """Raised when no configured LLM provider is reachable."""


class LLMResponseError(RuntimeError):
    """Raised when a provider returns an error status or an empty completion."""


class LLMBadJSONError(ValueError):
    """Raised when a model response could not be parsed as JSON after retries."""


@dataclass
class LLMResponse:
    """A completed generation with provenance."""

    content: str
    model: str
    provider: str
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None

    def text(self) -> str:
        """Raw completion text."""
        return self.content


def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
    """Build an OpenAI-style message list."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating fences and leading prose.

    Args:
        text: Raw model output.

    Returns:
        dict[str, Any]: Parsed object.

    Raises:
        LLMBadJSONError: No parseable JSON object was found.
    """
    from app.utils.json import extract_json_object

    return extract_json_object(text)


class LLMService:
    """Unified chat/completions client for OpenRouter and Ollama."""

    def __init__(self, *, timeout_seconds: float | None = None) -> None:
        self.timeout = timeout_seconds or settings.llm_timeout_seconds
        self._client: httpx.AsyncClient | None = None
        self._last_probe: dict[str, Any] = {}

    # ------------------------------------------------------------------ provider selection
    @property
    def provider(self) -> str:
        """Active provider name: ``openrouter``, ``ollama`` or ``none``."""
        if settings.openrouter_api_key:
            return "openrouter"
        if settings.ollama_enabled:
            return "ollama"
        return "none"

    @property
    def model(self) -> str:
        """Active model name for the selected provider."""
        return settings.openrouter_model if self.provider == "openrouter" else settings.ollama_model

    def available(self) -> bool:
        """Whether any provider is configured."""
        return self.provider != "none"

    async def _http(self) -> httpx.AsyncClient:
        """Shared async HTTP client (lazily created)."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close the shared HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def status(self) -> dict[str, Any]:
        """Provider availability report for the diagnostics endpoint."""
        report: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "openrouter_configured": bool(settings.openrouter_api_key),
            "ollama_enabled": settings.ollama_enabled,
        }
        if self.provider == "ollama":
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{settings.ollama_base_url}/api/tags")
                    report["reachable"] = response.status_code == 200
                    if response.status_code == 200:
                        tags = response.json().get("models") or []
                        report["models"] = [tag.get("name", "") for tag in tags][:20]
            except Exception as exc:  # noqa: BLE001 - probe failures are expected when offline
                report["reachable"] = False
                report["error"] = str(exc)[:200]
        elif self.provider == "openrouter":
            report["reachable"] = True
            report["base_url"] = settings.openrouter_base_url
        return report

    # ------------------------------------------------------------------ OpenRouter
    async def _openrouter(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> LLMResponse:
        """Non-streaming OpenRouter chat completion."""
        payload: dict[str, Any] = {
            "model": settings.openrouter_model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        client = await self._http()
        try:
            response = await client.post(
                f"{settings.openrouter_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "HTTP-Referer": settings.app_name,
                    "X-Title": settings.app_name,
                },
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"OpenRouter request failed: {exc}") from exc
        if response.status_code >= 400:
            raise LLMResponseError(f"OpenRouter HTTP {response.status_code}: {response.text[:400]}")
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        if not content.strip():
            raise LLMResponseError("OpenRouter returned an empty completion")
        return LLMResponse(
            content=content,
            model=data.get("model") or settings.openrouter_model,
            provider="openrouter",
            usage=data.get("usage") or {},
            finish_reason=choice.get("finish_reason"),
        )

    async def _openrouter_stream(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> AsyncIterator[str]:
        """Streaming OpenRouter chat completion yielding content deltas."""
        payload: dict[str, Any] = {
            "model": settings.openrouter_model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        client = await self._http()
        try:
            async with client.stream(
                "POST",
                f"{settings.openrouter_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.openrouter_api_key}",
                    "HTTP-Referer": settings.app_name,
                    "X-Title": settings.app_name,
                },
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise LLMResponseError(
                        f"OpenRouter HTTP {response.status_code}: {body.decode(errors='ignore')[:300]}"
                    )
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[len("data:"):].strip()
                    if chunk == "[DONE]":
                        return
                    try:
                        data = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0].get("delta") or {}).get("content")
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"OpenRouter stream failed: {exc}") from exc

    # ------------------------------------------------------------------ Ollama
    async def _ollama(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> LLMResponse:
        """Non-streaming Ollama chat completion."""
        payload: dict[str, Any] = {
            "model": settings.ollama_model,
            "messages": list(messages),
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": settings.ollama_num_ctx,
            },
        }
        if json_mode:
            payload["format"] = "json"
        client = await self._http()
        try:
            response = await client.post(f"{settings.ollama_base_url}/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"Ollama request failed: {exc}") from exc
        if response.status_code >= 400:
            raise LLMResponseError(f"Ollama HTTP {response.status_code}: {response.text[:400]}")
        data = response.json()
        content = (data.get("message") or {}).get("content") or ""
        if not content.strip():
            raise LLMResponseError("Ollama returned an empty completion")
        return LLMResponse(
            content=content,
            model=data.get("model") or settings.ollama_model,
            provider="ollama",
            usage={
                "prompt_eval_count": data.get("prompt_eval_count"),
                "eval_count": data.get("eval_count"),
            },
            finish_reason=data.get("done_reason"),
        )

    async def _ollama_stream(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> AsyncIterator[str]:
        """Streaming Ollama chat completion yielding content deltas."""
        payload: dict[str, Any] = {
            "model": settings.ollama_model,
            "messages": list(messages),
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": settings.ollama_num_ctx,
            },
        }
        if json_mode:
            payload["format"] = "json"
        client = await self._http()
        try:
            async with client.stream(
                "POST", f"{settings.ollama_base_url}/api/chat", json=payload
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise LLMResponseError(
                        f"Ollama HTTP {response.status_code}: {body.decode(errors='ignore')[:300]}"
                    )
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    delta = (data.get("message") or {}).get("content")
                    if delta:
                        yield delta
                    if data.get("done"):
                        return
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"Ollama stream failed: {exc}") from exc

    # ------------------------------------------------------------------ public API
    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate a completion with the active provider.

        Args:
            prompt: User prompt.
            system: Optional system instruction.
            max_tokens: Output token cap.
            temperature: Sampling temperature.
            json_mode: Request JSON-constrained output.

        Returns:
            LLMResponse: Completion with provider provenance.

        Raises:
            LLMUnavailableError: No provider configured or provider unreachable.
        """
        messages = _messages(prompt, system)
        max_tokens = max_tokens or settings.llm_max_tokens
        temperature = settings.llm_temperature if temperature is None else temperature

        if self.provider == "openrouter":
            return await self._openrouter(
                messages, max_tokens=max_tokens, temperature=temperature, json_mode=json_mode
            )
        if self.provider == "ollama":
            return await self._ollama(
                messages, max_tokens=max_tokens, temperature=temperature, json_mode=json_mode
            )
        raise LLMUnavailableError("No LLM provider configured (set OPENROUTER_API_KEY or OLLAMA_ENABLED=1)")

    async def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> AsyncIterator[str]:
        """Stream completion deltas from the active provider."""
        messages = _messages(prompt, system)
        max_tokens = max_tokens or settings.llm_max_tokens
        temperature = settings.llm_temperature if temperature is None else temperature

        if self.provider == "openrouter":
            async for delta in self._openrouter_stream(
                messages, max_tokens=max_tokens, temperature=temperature, json_mode=json_mode
            ):
                yield delta
            return
        if self.provider == "ollama":
            async for delta in self._ollama_stream(
                messages, max_tokens=max_tokens, temperature=temperature, json_mode=json_mode
            ):
                yield delta
            return
        raise LLMUnavailableError("No LLM provider configured (set OPENROUTER_API_KEY or OLLAMA_ENABLED=1)")

    async def generate_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_attempts: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Generate a JSON object, repairing malformed output with bounded retries.

        Args:
            prompt: User prompt (schema should be described in the prompt).
            system: Optional system instruction.
            max_attempts: Parse retries (defaults to ``settings.json_retry_attempts``).
            max_tokens: Output token cap.
            temperature: Sampling temperature (low for structured output).

        Returns:
            dict[str, Any]: Parsed JSON object.

        Raises:
            LLMBadJSONError: All attempts failed to produce parseable JSON.
            LLMUnavailableError: No provider available.
        """
        attempts = max_attempts if max_attempts is not None else settings.json_retry_attempts
        current_prompt = prompt
        errors: list[str] = []
        last_error: Exception | None = None

        for attempt in range(max(1, attempts)):
            json_mode = self.provider == "ollama"
            try:
                response = await self.generate(
                    current_prompt,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=0.0 if temperature is None else temperature,
                    json_mode=json_mode,
                )
            except LLMUnavailableError:
                raise
            except LLMResponseError as exc:
                last_error = exc
                errors.append(str(exc))
                current_prompt = prompt + "\n\nPrevious attempt failed: " + str(exc)
                continue

            try:
                return _extract_json(response.content)
            except ValueError as exc:
                last_error = exc
                errors.append(str(exc))
                logger.warning(f"Attempt {attempt + 1}: invalid JSON from {response.provider}: {exc}")
                current_prompt = (
                    prompt
                    + "\n\nReturn ONLY a valid JSON object. Previous output was not parseable: "
                    + response.content[:600].replace("\n", " ")
                )
            await asyncio.sleep(min(1.5 * (attempt + 1), 5.0))

        raise LLMBadJSONError(
            f"Could not obtain valid JSON after {max(1, attempts)} attempt(s): {' | '.join(errors)[:400]}"
        ) from last_error


llm_service = LLMService()
