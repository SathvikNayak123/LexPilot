from fastapi import FastAPI

from api.middleware.cors import add_cors
from api.routes import health, chat, feedback
from config.config import settings

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
