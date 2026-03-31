from fastapi import FastAPI
from agents import set_tracing_disabled

from api.provider import openrouter_provider  # noqa: F401 — imported for side-effect availability
from api.middleware.cors import add_cors
from api.routes import health, chat, feedback
from config.config import settings

set_tracing_disabled(True)

app = FastAPI(
    title="LexPilot",
    description="Multi-agent legal intelligence system for Indian law",
    version="0.1.0",
)

# Middleware
add_cors(app)

# Routes
app.include_router(health.router)
app.include_router(chat.router)
app.include_router(feedback.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=settings.api_port, reload=True)
