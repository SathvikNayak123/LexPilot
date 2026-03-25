"""FastAPI application entry point for FinDocs RAG API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from findocs.api.routes.feedback import router as feedback_router
from findocs.api.routes.query import router as query_router
from findocs.config.config import get_settings

logger: structlog.stdlib.BoundLogger = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: initialise and tear down shared resources."""
    logger.info("findocs_api_starting", qdrant_url=settings.QDRANT_URL)
    yield
    logger.info("findocs_api_shutting_down")


app = FastAPI(
    title="FinDocs RAG API",
    description="Financial Document Intelligence for Indian Retail Investors",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router, prefix="/query", tags=["query"])
app.include_router(feedback_router, prefix="/feedback", tags=["feedback"])


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
