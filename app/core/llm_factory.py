"""LLM Factory for creating language model instances.

This module provides a centralized way to create and configure
DeepSeek LLM instances through its OpenAI-compatible API.
"""

from langchain_openai import ChatOpenAI

from app.config import settings


class LLMFactory:
    """Factory for creating LLM instances with consistent configuration."""

    @staticmethod
    def create_chat_model(
        model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatOpenAI:
        """
        Create a ChatOpenAI instance configured for DeepSeek.

        Args:
            model_name:
                DeepSeek model name.
                Uses configured default if None.

            temperature:
                Sampling temperature.
                Uses configured default if None.

            max_tokens:
                Maximum number of output tokens.
                Uses configured default if None.

        Returns:
            Configured ChatOpenAI instance connected to DeepSeek.

        Raises:
            ValueError:
                If DEEPSEEK_API_KEY is not configured.
        """

        if not settings.deepseek_api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is not configured. " "Please set it in your .env file."
            )

        return ChatOpenAI(
            model=model_name or settings.model_name,
            temperature=(temperature if temperature is not None else settings.model_temperature),
            max_tokens=(max_tokens if max_tokens is not None else settings.model_max_tokens),
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )


def get_llm() -> ChatOpenAI:
    """Get the default DeepSeek LLM instance."""
    return LLMFactory.create_chat_model()
