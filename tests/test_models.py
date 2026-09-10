from __future__ import annotations

import pytest

from claude_router.models import (
    ZAI_MODEL_IDS,
    ZAI_MODELS,
    catalog_input_modalities,
    compact_row,
    exact_models,
    input_modalities,
    namespaced_model,
    original_model,
    picker_row,
    provider_of,
    route_of_namespaced,
    search_models,
    supported_parameters,
    supports_parameter,
    supports_tools,
    tool_capability_badge,
)


def ids(models: list[dict[str, object]]) -> list[str]:
    return [str(model["id"]) for model in models]


def test_plain_search_is_case_insensitive_substring_glob(sample_models) -> None:
    result = search_models(sample_models, ["CLAUDE"])
    assert ids(result) == [
        "anthropic/claude-opus-4.6",
        "anthropic/claude-sonnet-4.6",
    ]


def test_shell_style_search_matches_each_metadata_field(sample_models) -> None:
    assert ids(search_models(sample_models, ["anthropic/*sonnet*"])) == [
        "anthropic/claude-sonnet-4.6"
    ]
    assert ids(search_models(sample_models, ["*coding model"])) == [
        "qwen/qwen3-coder",
        "anthropic/claude-sonnet-4.6",
    ]


def test_multiple_queries_are_or_patterns(sample_models) -> None:
    assert set(ids(search_models(sample_models, ["gemini", "qwen*"]))) == {
        "google/gemini-3.1-pro-preview",
        "qwen/qwen3-coder",
    }


def test_regex_search_and_error(sample_models) -> None:
    assert ids(search_models(sample_models, [r"^google/.+preview$"], regex=True)) == [
        "google/gemini-3.1-pro-preview"
    ]
    with pytest.raises(ValueError, match="invalid regular expression"):
        search_models(sample_models, ["["], regex=True)


def test_exact_models_preserves_order_and_rejects_unknown(sample_models) -> None:
    selected = exact_models(
        sample_models,
        ["qwen/qwen3-coder", "anthropic/claude-opus-4.6", "qwen/qwen3-coder"],
    )
    assert ids(selected) == ["qwen/qwen3-coder", "anthropic/claude-opus-4.6"]
    with pytest.raises(ValueError, match="not found"):
        exact_models(sample_models, ["missing/model"])


def test_picker_row_has_human_metadata_and_management_marker(sample_models) -> None:
    row = picker_row(sample_models[0])
    assert row["model"] == "anthropic/claude-sonnet-4.6"
    assert row["label"] == "Claude Sonnet 4.6"
    assert "OpenRouter via claude-router" in row["description"]
    assert "$3/M input" in row["description"]
    assert "tools ?" in row["description"]


def test_tool_capabilities_are_explicit_and_unknown_is_not_assumed(sample_models) -> None:
    gemini = sample_models[2]
    qwen = sample_models[3]
    unknown = sample_models[0]

    assert supported_parameters(gemini) == frozenset({"tools", "tool_choice", "max_tokens"})
    assert supports_parameter(gemini, "TOOLS") is True
    assert supports_tools(gemini) is True
    assert tool_capability_badge(gemini, detailed=True) == "tools ✓ · tool choice ✓"
    assert supports_parameter(qwen, "tools") is False
    assert supports_tools(qwen) is False
    assert tool_capability_badge(qwen, detailed=True) == "tools ✗ · tool choice ✗"
    assert supports_parameter(unknown, "tools") is None
    assert supports_tools(unknown) is False
    assert tool_capability_badge(unknown) == "tools ?"


def test_compact_row_exposes_tool_metadata(sample_models) -> None:
    assert compact_row(sample_models[2]).endswith("\t✓\t✓")
    assert compact_row(sample_models[3]).endswith("\t✗\t✗")
    assert compact_row(sample_models[0]).endswith("\t?\t?")


def test_catalog_input_modalities_uses_exact_ids_and_skips_unknown_metadata() -> None:
    models = [
        {
            "id": "text/model",
            "architecture": {"input_modalities": ["Text"]},
        },
        {
            "id": "vision/model",
            "architecture": {"input_modalities": ["text", "IMAGE", "video"]},
        },
        {"id": "unknown/model"},
    ]

    assert input_modalities(models[0]) == frozenset({"text"})
    assert input_modalities(models[2]) is None
    assert catalog_input_modalities(models) == {
        "text/model": frozenset({"text"}),
        "vision/model": frozenset({"text", "image", "video"}),
    }


def test_zai_catalog_is_static_text_only_and_tool_capable() -> None:
    assert {str(model["id"]) for model in ZAI_MODELS} == {
        "glm-5.3",
        "glm-5.3-flash",
        "glm-5.3-highspeed",
        "glm-5.2",
        "glm-5-turbo",
        "glm-4.7",
    }
    assert {str(model["id"]) for model in ZAI_MODELS} == ZAI_MODEL_IDS
    for model in ZAI_MODELS:
        assert model["provider"] == "zai"
        assert model["supported_parameters"] == ["tools", "tool_choice"]
        assert model["architecture"] == {"input_modalities": ["text"]}
        assert isinstance(model["context_length"], int)
        assert "pricing" not in model


def test_provider_of_and_namespacing_round_trip() -> None:
    assert provider_of("glm-5.3-flash") == "zai"
    assert provider_of("z-ai/glm-5.3-flash") == "openrouter"
    assert namespaced_model("glm-5.3-flash") == "clr/zai/glm-5.3-flash"
    assert namespaced_model("z-ai/glm-5.3-flash") == "clr/openrouter/z-ai/glm-5.3-flash"
    assert original_model("clr/zai/glm-5.3-flash") == "glm-5.3-flash"
    assert original_model("clr/openrouter/z-ai/glm-5.3-flash") == "z-ai/glm-5.3-flash"
    assert original_model("claude-opus-4-8") is None
    assert original_model("clr/zai/") is None


def test_route_of_namespaced_distinguishes_prefixes() -> None:
    assert route_of_namespaced("clr/zai/glm-5.3") == "zai"
    assert route_of_namespaced("clr/openrouter/z-ai/glm-5.3") == "openrouter"
    assert route_of_namespaced("glm-5.3") is None
    assert route_of_namespaced("clr/other/glm-5.3") is None


def test_zai_picker_row_labels_and_describe_the_coding_plan_without_pricing() -> None:
    glm = next(model for model in ZAI_MODELS if model["id"] == "glm-5.3-flash")
    row = picker_row(glm, hybrid=True)

    assert row["model"] == "clr/zai/glm-5.3-flash[1m]"
    assert row["label"] == "GLM-5.3 Flash · Z.ai"
    assert "Z.ai Coding Plan via claude-router" in row["description"]
    assert "$" not in row["description"]
    assert "1000K context" in row["description"]

    hybrid = picker_row(glm)
    assert hybrid["model"] == "glm-5.3-flash"
    assert hybrid["label"] == "GLM-5.3 Flash"
