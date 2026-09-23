from functools import lru_cache

from langchain_anthropic import ChatAnthropic

from app.config import get_settings


@lru_cache
def get_sonnet() -> ChatAnthropic:
    settings = get_settings()
    # No `temperature` override: the Claude 5 family rejects it as a deprecated param.
    return ChatAnthropic(model=settings.sonnet_model, anthropic_api_key=settings.anthropic_api_key)


@lru_cache
def get_haiku() -> ChatAnthropic:
    settings = get_settings()
    return ChatAnthropic(model=settings.haiku_model, anthropic_api_key=settings.anthropic_api_key)
