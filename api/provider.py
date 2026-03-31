from openai import AsyncOpenAI
from agents.models.multi_provider import MultiProvider

from config.config import settings

openrouter_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.openrouter_api_key,
    default_headers={
        "HTTP-Referer": "https://lexpilot.ai",
        "X-Title": "LexPilot",
    },
)

openrouter_provider = MultiProvider(
    openai_client=openrouter_client,
    openai_use_responses=False,
    unknown_prefix_mode="model_id",
)
