"""Shared LLM initialization logic."""

from .config import settings
from pydantic import SecretStr

def get_llm(temperature: float = 0.0):
    """Get the configured LLM instance with the specified temperature."""
    if settings.LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(  # type: ignore[call-arg]
            model_name=settings.effective_model,
            api_key=SecretStr(settings.ANTHROPIC_API_KEY) if settings.ANTHROPIC_API_KEY else SecretStr(""),
            temperature=temperature,
        )
    elif settings.LLM_PROVIDER == "azure":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=SecretStr(settings.AZURE_OPENAI_API_KEY) if settings.AZURE_OPENAI_API_KEY else None,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_deployment=settings.effective_model,
            temperature=temperature,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.effective_model,
            api_key=SecretStr(settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None,
            temperature=temperature,
        )
