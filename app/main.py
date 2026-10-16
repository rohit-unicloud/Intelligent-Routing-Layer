
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

# Load project-root .env before importing the router.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.router import set_http_client, route_request

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    timeout = httpx.Timeout(
        connect=3.0,
        read=float(os.getenv("ROUTER_LLM_TIMEOUT_SECONDS", "8")),
        write=5.0,
        pool=3.0,
    )

    limits = httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=30.0,
    )

    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
    ) as client:
        set_http_client(client)
        logger.info("Routing service HTTP client initialized")

        try:
            yield
        finally:
            set_http_client(None)
            logger.info("Routing service shutting down")


app = FastAPI(
    title="Intelligent Routing Layer",
    description=(
        "Classifies requests and returns a validated destination. "
        "Does not execute inference."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


class Requirements(BaseModel):
    quality: Literal["low", "medium", "high"] | None = None
    latency: Literal["low", "normal", "relaxed"] | None = None
    cost: Literal["low", "medium", "high"] | None = None
    modalities: list[str] = Field(default_factory=list)
    output_format: str | None = None


class RouteRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=100000)
    requirements: Requirements = Field(default_factory=Requirements)
    metadata: dict[str, Any] = Field(default_factory=dict)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "intelligent-routing-layer",
    }


@app.post("/route")
async def route(body: RouteRequest):
    try:
        return await route_request(
            prompt=body.prompt,
            requirements=body.requirements.model_dump(exclude_none=True),
            metadata=body.metadata,
        )
    except ValueError as exc:
        logger.warning("Invalid routing request or configuration: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Routing service runtime error")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
