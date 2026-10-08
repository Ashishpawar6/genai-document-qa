"""The language model that writes answers. Optional: without an API key the app is search-only."""
from langchain_anthropic import ChatAnthropic

from docqa.config import Settings


def get_llm(settings: Settings) -> ChatAnthropic | None:
    """Return a Claude chat model, or None when no API key is configured.

    No temperature / top_p / top_k are sent: the newest Claude models reject non-default sampling
    values. `max_tokens` is set explicitly because thinking tokens count toward it, and the
    connector's own default is far too large for a normal (non-streaming) request.
    """
    if not settings.has_api_key:
        return None
    kwargs = {}
    if settings.effort:
        kwargs["effort"] = settings.effort
    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        max_tokens=settings.max_tokens,
        **kwargs,
    )
