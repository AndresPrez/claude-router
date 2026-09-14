"""Tests for the Databricks AI Gateway bridge (Responses API dialect)."""

import json

import pytest

from claude_router.databricks import (
    DatabricksBridgeError,
    translate_request,
    translate_response,
    translate_stream_events,
    validate_base_url,
)


def test_translate_request_maps_core_fields() -> None:
    payload = {
        "model": "system.ai.glm-5-3-flash",
        "max_tokens": 512,
        "system": "You are terse.",
        "temperature": 0.2,
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
            {"role": "user", "content": "bye"},
        ],
        "tools": [
            {
                "name": "grep",
                "description": "search",
                "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
        ],
    }

    body = translate_request(payload, "high")

    assert body["model"] == "system.ai.glm-5-3-flash"
    assert body["instructions"] == "You are terse."
    assert body["max_output_tokens"] == 512
    assert body["temperature"] == 0.2
    assert body["reasoning"] == {"effort": "high"}
    assert body["tools"] == [{
        "type": "function", "name": "grep", "description": "search",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    }]
    kinds = [(i.get("role"), i["content"][0]["type"]) for i in body["input"]]
    assert kinds == [
        ("user", "input_text"),
        ("assistant", "output_text"),
        ("user", "input_text"),
    ]


def test_translate_request_maps_tool_roundtrip() -> None:
    payload = {
        "model": "m",
        "messages": [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "grep", "input": {"q": "x"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1",
                 "content": [{"type": "text", "text": "match"}]},
            ]},
        ],
    }

    body = translate_request(payload, None)
    call, output = body["input"]

    assert call == {"type": "function_call", "call_id": "t1", "name": "grep",
                    "arguments": json.dumps({"q": "x"})}
    assert output == {"type": "function_call_output", "call_id": "t1", "output": "match"}


def test_translate_response_text_and_tools() -> None:
    document = {
        "id": "resp_1",
        "output": [
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "Running it."}]},
            {"type": "function_call", "call_id": "call_9", "name": "grep",
             "arguments": "{\"q\": \"z\"}"},
        ],
        "usage": {"input_tokens": 100, "output_tokens": 40},
    }

    response = translate_response(document, "system.ai.glm-5-3-flash")

    assert response["content"][0] == {"type": "text", "text": "Running it."}
    assert response["content"][1]["type"] == "tool_use"
    assert response["content"][1]["input"] == {"q": "z"}
    assert response["stop_reason"] == "tool_use"
    assert response["usage"]["input_tokens"] == 100


def test_translate_stream_events_to_anthropic_frames() -> None:
    events = iter([
        {"type": "response.created", "payload": {}},
        {"type": "response.output_item.added",
         "payload": {"item": {"type": "message"}}},
        {"type": "response.output_text.delta", "payload": {"delta": "Hel"}},
        {"type": "response.output_text.delta", "payload": {"delta": "lo"}},
        {"type": "response.output_item.done", "payload": {}},
        {"type": "response.completed",
         "payload": {"response": {"usage": {"input_tokens": 12, "output_tokens": 2}}}},
    ])

    frames = [f.decode() for f in translate_stream_events(events, "m")]
    kinds = [f.splitlines()[0].removeprefix("event: ") for f in frames]

    assert kinds == [
        "message_start", "content_block_start",
        "content_block_delta", "content_block_delta",
        "content_block_stop", "message_delta", "message_stop",
    ]
    usage_frame = frames[-2]
    assert '"output_tokens":2' in usage_frame


def test_stream_failures_raise() -> None:
    with pytest.raises(DatabricksBridgeError, match="no events"):
        list(translate_stream_events(iter([]), "m"))
    with pytest.raises(DatabricksBridgeError, match="without completion"):
        list(translate_stream_events(iter([{"type": "response.created", "payload": {}}]), "m"))


def test_validate_base_url() -> None:
    assert validate_base_url("https://gw.example.com/").endswith(".com")
    with pytest.raises(DatabricksBridgeError):
        validate_base_url("http://insecure.example.com")
