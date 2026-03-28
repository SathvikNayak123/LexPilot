from agents.extensions.models.litellm_model import LitellmModel

from config.config import settings


def get_tier2_model():
    """Tier 2: Groq LLaMA 3.3 70B — single-agent, moderate complexity."""
    return LitellmModel(
        model="groq/llama-3.3-70b-versatile",
        api_key=settings.groq_api_key,
    )


def get_tier3_model():
    """Tier 3: Claude Sonnet — complex reasoning, orchestration, synthesis."""
    return LitellmModel(
        model="anthropic/claude-sonnet-4-20250514",
        api_key=settings.anthropic_api_key,
    )
