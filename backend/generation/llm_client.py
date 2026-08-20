"""LLM client using LangChain's ChatOpenAI with OpenRouter backend."""

from collections.abc import AsyncGenerator

from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.core.exceptions import GenerationError
from backend.core.logging import get_logger

logger = get_logger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class LLMClient:
    """LLM client using LangChain's ChatOpenAI pointed at OpenRouter.

    Provides both synchronous generation and async streaming via
    LangChain's unified interface, with OpenRouter as the backend.

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
        """Initialize the LLM client.

        Args:
            model_name: OpenRouter model identifier.
            api_token: OpenRouter API key.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._api_token = api_token
        self._llm: ChatOpenAI | None = None

    def _get_llm(self, model_override: str | None = None) -> ChatOpenAI:
        """Get or create a ChatOpenAI instance.

        Args:
            model_override: Optional model name to override the default.

        Returns:
            Configured ChatOpenAI instance.
        """
        model = model_override or self.model_name

        # If using default model, cache the instance
        if model == self.model_name:
            if self._llm is None:
                self._llm = ChatOpenAI(
                    model=self.model_name,
                    openai_api_key=self._api_token,
                    openai_api_base=OPENROUTER_BASE_URL,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    default_headers={
                        "HTTP-Referer": "https://github.com/luma-ai",
                        "X-Title": "Luma",
                    },
                )
            return self._llm

        # For overridden models, create a new instance
        return ChatOpenAI(
            model=model,
            openai_api_key=self._api_token,
            openai_api_base=OPENROUTER_BASE_URL,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            default_headers={
                "HTTP-Referer": "https://github.com/luma-ai",
                "X-Title": "Luma",
            },
        )

    @property
    def llm(self) -> ChatOpenAI:
        """Get the default ChatOpenAI instance.

        Returns:
            The cached ChatOpenAI instance for the default model.
        """
        return self._get_llm()

    def _build_messages(self, system_prompt: str, user_prompt: str) -> list[BaseMessage]:
        """Build a message list from system and user prompts.

        Args:
            system_prompt: System instructions for the model.
            user_prompt: User message with context and query.

        Returns:
            List of LangChain message objects.
        """
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

    async def generate(self, system_prompt: str, user_prompt: str, model_override: str | None = None) -> str:
        """Generate a complete response from the LLM.

        Args:
            system_prompt: System instructions for the model.
            user_prompt: User message with context and query.
            model_override: Optional model name to use instead of default.

        Returns:
            The generated text response.

        Raises:
            GenerationError: If the API call fails.
        """
        messages = self._build_messages(system_prompt, user_prompt)
        llm = self._get_llm(model_override)

        try:
            response = await llm.ainvoke(messages)
            content = response.content
            if isinstance(content, str):
                return content.strip()
            return str(content).strip()

        except Exception as exc:
            error_msg = str(exc)
            if "402" in error_msg:
                raise GenerationError("OpenRouter credits exhausted or payment required.") from exc
            if "429" in error_msg:
                raise GenerationError("Rate limit reached. Please wait a moment and try again.") from exc
            raise GenerationError(f"LLM generation failed: {error_msg}") from exc

    async def generate_stream(
        self, system_prompt: str, user_prompt: str, model_override: str | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream generated tokens from the LLM.

        Args:
            system_prompt: System instructions for the model.
            user_prompt: User message with context and query.
            model_override: Optional model name to use instead of default.

        Yields:
            Text chunks as they are generated.

        Raises:
            GenerationError: If the API call fails.
        """
        messages = self._build_messages(system_prompt, user_prompt)
        llm = self._get_llm(model_override)

        try:
            async for chunk in llm.astream(messages):
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    content = chunk.content
                    if isinstance(content, str):
                        yield content
                    else:
                        yield str(content)

        except GenerationError:
            raise
        except Exception as exc:
            error_msg = str(exc)
            if "402" in error_msg:
                raise GenerationError("OpenRouter credits exhausted or payment required.") from exc
            if "429" in error_msg:
                raise GenerationError("Rate limit reached. Please wait a moment and try again.") from exc
            raise GenerationError(f"Streaming generation failed: {error_msg}") from exc
