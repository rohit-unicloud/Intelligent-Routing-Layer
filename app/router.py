
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import yaml

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(os.getenv("ROUTER_CONFIG", "config/models.yaml"))

# One reusable HTTP client per application worker.
_http_client: httpx.AsyncClient | None = None


def set_http_client(client: httpx.AsyncClient | None) -> None:
    global _http_client
    _http_client = client


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict) or not isinstance(config.get("routes"), dict):
        raise ValueError("Invalid routing configuration: expected a routes mapping")

    return config


def _route_catalog(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return config["routes"]


def _fallback_task(prompt: str) -> str:
    """Deterministic fallback used when LLM classification is unavailable."""
    p = prompt.lower()

    if any(x in p for x in (
        "generate an image", "create an image", "edit this image",
        "image generation",
    )):
        return "image_generation"

    if any(x in p for x in (
        "generate a video", "image to video", "edit this video",
        "video generation",
    )):
        return "video_generation_editing"

    if any(x in p for x in (
        "cinematic", "film shot", "movie scene", "cinematic video",
    )):
        return "cinematic_video"

    if any(x in p for x in (
        "powerpoint", "pptx", "word document", "excel file",
        "create a report", "generate a document",
    )):
        return "document_generation"

    if any(x in p for x in (
        "analyze this dataset", "analyse this dataset", "data analysis",
        "csv file", "spreadsheet analysis",
    )):
        return "data_analysis"

    if any(x in p for x in (
        "debug", "write code", "python function", "repository",
        "refactor", "software development",
    )):
        return "coding"

    if any(x in p for x in (
        "search the web", "latest information", "research",
        "compare sources", "find sources",
    )):
        return "web_research"

    if any(x in p for x in (
        "plan a multi-step", "complex reasoning",
        "design an architecture", "analyze this architecture",
    )):
        return "advanced_reasoning"

    return "general_chat"


async def _classify_with_llm(
    prompt: str,
    requirements: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    api_key = os.getenv("ROUTER_LLM_API_KEY")
    base_url = os.getenv("ROUTER_LLM_BASE_URL", "").rstrip("/")
    model = os.getenv("ROUTER_LLM_MODEL", "")

    if not (api_key and base_url and model):
        logger.warning(
            "LLM classification disabled: one or more ROUTER_LLM settings are missing"
        )
        return None

    if _http_client is None:
        raise RuntimeError("HTTP client is not initialized")

    allowed = list(_route_catalog(config).keys())

    system = (
        "You are a request classifier for an AI routing layer. Do not answer "
        "the user's request. Choose exactly one task from allowed_tasks. Never "
        "invent a destination or model. Return only JSON with keys: task, "
        "complexity, modalities, quality_priority, latency_priority, "
        "cost_priority, confidence, reason. task must be one of allowed_tasks. "
        "complexity must be low, medium, or high. modalities must be a list "
        "of strings. confidence must be a number from 0 to 1. If the user asks "
        "to create or edit a video, choose video_generation_editing unless "
        "they explicitly ask for a cinematic or high-fidelity film-like result, "
        "in which case choose cinematic_video. Use document_generation for "
        "creating or editing files, data_analysis for interpreting datasets, "
        "coding for code or repository work, and web_research for web research."
    )

    user = {
        "allowed_tasks": allowed,
        "prompt": prompt,
        "user_requirements": requirements,
    }

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(user, ensure_ascii=False),
            },
        ],
        "response_format": {"type": "json_object"},
    }

    url = base_url
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"

    response = await _http_client.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    response.raise_for_status()

    body = response.json()
    content = body["choices"][0]["message"]["content"]

    if isinstance(content, list):
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        )

    parsed = json.loads(content)

    if not isinstance(parsed, dict) or parsed.get("task") not in allowed:
        logger.warning("LLM returned an invalid classification")
        return None

    try:
        confidence = float(parsed.get("confidence", 0.0))
        parsed["confidence"] = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        parsed["confidence"] = 0.0

    if not isinstance(parsed.get("modalities", []), list):
        parsed["modalities"] = []

    return parsed


async def route_request(
    prompt: str,
    requirements: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    config = load_config()
    routes = _route_catalog(config)

    try:
        analysis = await _classify_with_llm(prompt, requirements, config)
        source = "llm" if analysis is not None else "fallback"

    except (
        httpx.HTTPError,
        KeyError,
        ValueError,
        TypeError,
    ):
        logger.exception(
            "LLM classification failed; using deterministic fallback"
        )
        analysis = None
        source = "fallback"

    if analysis is None:
        task = _fallback_task(prompt)
        analysis = {
            "task": task,
            "complexity": "medium",
            "modalities": requirements.get("modalities", []),
            "quality_priority": requirements.get("quality", "medium"),
            "latency_priority": requirements.get("latency", "normal"),
            "cost_priority": requirements.get("cost", "medium"),
            "confidence": 0.55,
            "reason": "Deterministic fallback classification.",
        }
        source = "fallback"

    task = analysis["task"]

    if task not in routes:
        logger.error("Classifier selected an unknown route: %s", task)
        task = _fallback_task(prompt)
        analysis["reason"] = "Unknown route returned; deterministic fallback used."
        source = "fallback"

        if task not in routes:
            task = "general_chat"

        analysis["task"] = task

    destination = routes[task]

    modalities = {
        str(x).lower() for x in analysis.get("modalities", [])
    }
    modalities.update(
        str(x).lower() for x in requirements.get("modalities", [])
    )

    if "image" in modalities and task == "general_chat":
        task = "advanced_reasoning"

        if task not in routes:
            task = "general_chat"
        else:
            analysis["reason"] = (
                "Image input was declared; selected the configured "
                "multimodal reasoning destination."
            )
            analysis["confidence"] = min(
                float(analysis.get("confidence", 0.0)), 0.6
            )

        destination = routes[task]

    return {
        "route": task,
        "destination_type": destination["destination_type"],
        "destination_id": destination["destination_id"],
        "capability": destination.get("capability"),
        "analysis": {
            "complexity": analysis.get("complexity", "medium"),
            "modalities": sorted(modalities),
            "quality_priority": analysis.get(
                "quality_priority", requirements.get("quality", "medium")
            ),
            "latency_priority": analysis.get(
                "latency_priority", requirements.get("latency", "normal")
            ),
            "cost_priority": analysis.get(
                "cost_priority", requirements.get("cost", "medium")
            ),
            "confidence": analysis.get("confidence", 0.0),
            "reason": analysis.get("reason", ""),
            "classifier": source,
        },
        "metadata": metadata,
    }
