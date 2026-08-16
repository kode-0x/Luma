"""LLM client for text generation via OpenRouter API."""

import json
from collections.abc import AsyncGenerator

import httpx

from backend.core.exceptions import GenerationError
from backend.core.logging import get_logger

logger = get_logger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMClient:
    """Client for generating text using the OpenRouter API.

    OpenRouter provides a unified OpenAI-compatible interface to many
    models, including free-tier options.

    Attributes:
        model_name: OpenRouter model identifier.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (lower = more deterministic).
    """

    def __init__(
        self,
        model_name: str,
        api_token: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._api_token = api_token

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_token}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/luma-ai",
            "X-Title": "Luma",
        }

    def _build_payload(self, system_prompt: str, user_prompt: str, stream: bool = False, model_override: str | None = None) -> dict:
        return {
            "model": model_override or self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": stream,
        }

    async def generate(self, system_prompt: str, user_prompt: str, model_override: str | None = None) -> str:
        """Generate a complete response from the LLM.

        Args:
            system_prompt: System instructions for the model.
            user_prompt: User message with context and query.

        Returns:
            The generated text response.

        Raises:
            GenerationError: If the API call fails.
        """
        payload = self._build_payload(system_prompt, user_prompt, stream=False, model_override=model_override)

        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(
                    OPENROUTER_URL,
                    headers=self._build_headers(),
                    json=payload,
                )

                if response.status_code != 200:
                    error_detail = response.text[:500]
                    if response.status_code == 402:
                        raise GenerationError("OpenRouter credits exhausted or payment required.")
                    if response.status_code == 429:
                        raise GenerationError("Rate limit reached. Please wait a moment and try again.")
                    raise GenerationError(
                        f"LLM API returned status {response.status_code}: {error_detail}"
                    )

                result = response.json()
                choices = result.get("choices", [])
                if not choices:
                    raise GenerationError("LLM returned empty choices")

                return choices[0].get("message", {}).get("content", "").strip()

        except httpx.TimeoutException as exc:
            raise GenerationError("LLM API request timed out") from exc
        except httpx.HTTPError as exc:
            raise GenerationError(f"HTTP error during LLM generation: {exc}") from exc
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(f"Unexpected error during generation: {exc}") from exc

    async def generate_stream(self, system_prompt: str, user_prompt: str, model_override: str | None = None) -> AsyncGenerator[str, None]:
        """Stream generated tokens from the LLM.

        Args:
            system_prompt: System instructions for the model.
            user_prompt: User message with context and query.

        Yields:
            Text chunks as they are generated.

        Raises:
            GenerationError: If the API call fails.
        """
        payload = self._build_payload(system_prompt, user_prompt, stream=True, model_override=model_override)

        try:
            async with httpx.AsyncClient(timeout=120.0) as client, client.stream(
                "POST",
                OPENROUTER_URL,
                headers=self._build_headers(),
                json=payload,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise GenerationError(
                        f"LLM streaming API returned status {response.status_code}: {body.decode()[:500]}"
                    )

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue

                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue

        except GenerationError:
            raise
        except httpx.TimeoutException as exc:
            raise GenerationError("LLM streaming request timed out") from exc
        except Exception as exc:
            raise GenerationError(f"Streaming generation failed: {exc}") from exc
