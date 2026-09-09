"""Model matching, ranking, and display helpers."""

from __future__ import annotations

import fnmatch
import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any

OPENROUTER_MODEL_PREFIX = "clr/openrouter/"
ZAI_MODEL_PREFIX = "clr/zai/"

# Static Z.ai Coding Plan catalog. OpenRouter model ids always contain a
# slash, so these slash-free GLM ids can never collide with that namespace.
ZAI_MODELS: list[dict[str, Any]] = [
    {
        "id": "glm-5.3",
        "name": "GLM-5.3",
        "description": "Flagship GLM coding model with the deepest reasoning",
        "provider": "zai",
        "context_length": 1_000_000,
        "supported_parameters": ["tools", "tool_choice"],
        "architecture": {"input_modalities": ["text"]},
    },
    {
        "id": "glm-5.3-flash",
        "name": "GLM-5.3 Flash",
        "description": "Fast, low-latency GLM coding model for everyday tasks",
        "provider": "zai",
        "context_length": 1_000_000,
        "supported_parameters": ["tools", "tool_choice"],
        "architecture": {"input_modalities": ["text"]},
    },
    {
        "id": "glm-5.3-highspeed",
        "name": "GLM-5.3 Highspeed",
        "description": "High-speed GLM variant optimized for quick responses",
        "provider": "zai",
        "context_length": 1_000_000,
        "supported_parameters": ["tools", "tool_choice"],
        "architecture": {"input_modalities": ["text"]},
    },
    {
        "id": "glm-5.2",
        "name": "GLM-5.2",
        "description": "Previous-generation GLM flagship with dependable coding quality",
        "provider": "zai",
        "context_length": 1_000_000,
        "supported_parameters": ["tools", "tool_choice"],
        "architecture": {"input_modalities": ["text"]},
    },
    {
        "id": "glm-5-turbo",
        "name": "GLM-5 Turbo",
        "description": "Turbo GLM model balancing speed and capability",
        "provider": "zai",
        "context_length": 200_000,
        "supported_parameters": ["tools", "tool_choice"],
        "architecture": {"input_modalities": ["text"]},
    },
    {
        "id": "glm-4.7",
        "name": "GLM-4.7",
        "description": "Compact GLM model for lighter coding workloads",
        "provider": "zai",
        "context_length": 200_000,
        "supported_parameters": ["tools", "tool_choice"],
        "architecture": {"input_modalities": ["text"]},
    },
]
ZAI_MODEL_IDS = frozenset(m["id"] for m in ZAI_MODELS)


def provider_of(model_id: str) -> str:
    """Return which route serves a bare catalog model id."""
    return "zai" if model_id in ZAI_MODEL_IDS else "openrouter"


def supported_parameters(model: dict[str, Any]) -> frozenset[str] | None:
    """Return normalized OpenRouter parameters, or ``None`` when unknown."""
    values = model.get("supported_parameters")
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        return None
    return frozenset(value.casefold() for value in values)


def supports_parameter(model: dict[str, Any], parameter: str) -> bool | None:
    parameters = supported_parameters(model)
    return None if parameters is None else parameter.casefold() in parameters


def supports_tools(model: dict[str, Any]) -> bool:
    return supports_parameter(model, "tools") is True


def _capability_mark(value: bool | None) -> str:
    if value is True:
        return "✓"
    if value is False:
        return "✗"
    return "?"


def tool_capability_badge(model: dict[str, Any], *, detailed: bool = False) -> str:
    tools = _capability_mark(supports_parameter(model, "tools"))
    if not detailed:
        return f"tools {tools}"
    tool_choice = _capability_mark(supports_parameter(model, "tool_choice"))
    return f"tools {tools} · tool choice {tool_choice}"


def input_modalities(model: dict[str, Any]) -> frozenset[str] | None:
    """Return normalized catalog input modalities, or ``None`` when unknown."""
    architecture = model.get("architecture")
    if not isinstance(architecture, dict):
        return None
    values = architecture.get("input_modalities")
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        return None
    return frozenset(value.casefold() for value in values)


def catalog_input_modalities(
    models: list[dict[str, Any]],
) -> dict[str, frozenset[str]]:
    """Index known input capabilities by exact OpenRouter model id."""
    result: dict[str, frozenset[str]] = {}
    for model in models:
        model_id = model.get("id")
        modalities = input_modalities(model)
        if isinstance(model_id, str) and modalities is not None:
            result[model_id] = modalities
    return result


def namespaced_model(model_id: str) -> str:
    prefix = ZAI_MODEL_PREFIX if provider_of(model_id) == "zai" else OPENROUTER_MODEL_PREFIX
    return f"{prefix}{model_id}"


def original_model(model_id: str) -> str | None:
    for prefix in (ZAI_MODEL_PREFIX, OPENROUTER_MODEL_PREFIX):
        if model_id.startswith(prefix):
            original = model_id[len(prefix) :]
            return original or None
    return None


def route_of_namespaced(model_id: str) -> str | None:
    """Return the route a namespaced model id addresses, or ``None``."""
    if model_id.startswith(ZAI_MODEL_PREFIX):
        return "zai"
    if model_id.startswith(OPENROUTER_MODEL_PREFIX):
        return "openrouter"
    return None


def hybrid_openrouter_allowed(model_id: str) -> bool:
    normalized = model_id.casefold()
    return not normalized.startswith("anthropic/") and normalized != "openrouter/auto"


def searchable_text(model: dict[str, Any]) -> str:
    values = (model.get("id"), model.get("name"), model.get("description"))
    return "\n".join(value for value in values if isinstance(value, str))


def searchable_fields(model: dict[str, Any]) -> list[str]:
    values = (model.get("id"), model.get("name"), model.get("description"))
    return [value for value in values if isinstance(value, str)]


def _glob_pattern(query: str) -> str:
    return query if any(marker in query for marker in "*?[") else f"*{query}*"


def search_models(
    models: list[dict[str, Any]], queries: list[str], *, regex: bool = False
) -> list[dict[str, Any]]:
    if not queries:
        return list(models)
    if regex:
        try:
            patterns = [re.compile(query, re.IGNORECASE) for query in queries]
        except re.error as exc:
            raise ValueError(f"invalid regular expression: {exc}") from exc

        def matches(model: dict[str, Any]) -> bool:
            fields = searchable_fields(model)
            return any(
                pattern.search(field) is not None for field in fields for pattern in patterns
            )

    else:
        patterns = [_glob_pattern(query).casefold() for query in queries]

        def matches(model: dict[str, Any]) -> bool:
            fields = [field.casefold() for field in searchable_fields(model)]
            return any(
                fnmatch.fnmatchcase(field, pattern) for field in fields for pattern in patterns
            )

    found = [model for model in models if matches(model)]
    return sorted(found, key=lambda model: _rank(model, queries))


def _rank(model: dict[str, Any], queries: list[str]) -> tuple[int, int, str]:
    model_id = str(model.get("id", "")).casefold()
    name = str(model.get("name", "")).casefold()
    plain = [query.casefold().strip("*?") for query in queries]
    score = 50
    for query in plain:
        if not query:
            continue
        if model_id == query:
            score = min(score, 0)
        elif name == query:
            score = min(score, 1)
        elif model_id.endswith(f"/{query}"):
            score = min(score, 2)
        elif query in model_id:
            score = min(score, 4 + model_id.index(query))
        elif query in name:
            score = min(score, 6 + name.index(query))
    return score, len(model_id), model_id


def top_matches(
    models: list[dict[str, Any]], query: str, *, limit: int = 15
) -> list[dict[str, Any]]:
    if not query.strip():
        return models[:limit]
    return search_models(models, [query])[:limit]


def exact_models(models: list[dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    by_id = {str(model["id"]): model for model in models}
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    seen: set[str] = set()
    for model_id in ids:
        if model_id in seen:
            continue
        seen.add(model_id)
        model = by_id.get(model_id)
        if model is None:
            missing.append(model_id)
        else:
            selected.append(model)
    if missing:
        rendered = ", ".join(missing)
        raise ValueError(f"model not found in the current model index: {rendered}")
    if not selected:
        raise ValueError("select at least one model")
    return selected


def _price_per_million(value: Any) -> str | None:
    try:
        price = Decimal(str(value)) * 1_000_000
    except (InvalidOperation, TypeError, ValueError):
        return None
    if price == 0:
        return "free"
    return f"${price.normalize():f}/M"


def picker_description(model: dict[str, Any]) -> str:
    zai = provider_of(str(model.get("id", ""))) == "zai"
    parts = [
        str(model.get("id", "")),
        "Z.ai Coding Plan via claude-router" if zai else "OpenRouter via claude-router",
        tool_capability_badge(model, detailed=True),
    ]
    context = model.get("context_length")
    if isinstance(context, int) and context > 0:
        parts.append(f"{context // 1000}K context" if context >= 1000 else f"{context} context")
    if not zai:
        pricing = model.get("pricing")
        if isinstance(pricing, dict):
            prompt = _price_per_million(pricing.get("prompt"))
            completion = _price_per_million(pricing.get("completion"))
            if prompt and completion:
                parts.append(f"{prompt} input · {completion} output")
    return " · ".join(parts)[:240]


def picker_row(model: dict[str, Any], *, hybrid: bool = False) -> dict[str, str]:
    model_id = str(model["id"])
    name = model.get("name")
    label = name if isinstance(name, str) and name else model_id
    suffix = " · Z.ai" if provider_of(model_id) == "zai" else " · OpenRouter"
    return {
        "model": namespaced_model(model_id) if hybrid else model_id,
        "label": f"{label}{suffix}" if hybrid else label,
        "description": picker_description(model),
    }


def compact_row(model: dict[str, Any]) -> str:
    model_id = str(model.get("id", ""))
    name = model.get("name")
    context = model.get("context_length")
    context_text = f"{context:,}" if isinstance(context, int) else "-"
    label = name if isinstance(name, str) else ""
    tools = _capability_mark(supports_parameter(model, "tools"))
    tool_choice = _capability_mark(supports_parameter(model, "tool_choice"))
    return f"{model_id}\t{label}\t{context_text}\t{tools}\t{tool_choice}"


def print_models(models: list[dict[str, Any]], *, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(models, indent=2, ensure_ascii=False))
        return
    print("MODEL\tNAME\tCONTEXT\tTOOLS\tTOOL_CHOICE")
    for model in models:
        print(compact_row(model))
