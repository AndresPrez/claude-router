"""Tests for the Cursor Cloud Agents bridge (clr/cursor/<model> route)."""

import json

import pytest

from claude_router import cursor as cursor_module
from claude_router.cursor import (
    AgentRegistry,
    CursorBridgeError,
    _images_from_last_user,
    _remember_conversation,
    _resolve_agent,
    bridge_frames,
    error_frame,
    flatten_conversation,
    run_messages,
)
from claude_router.models import (
    CURSOR_MODEL_PREFIX,
    namespaced_model,
    original_model,
    picker_row,
    provider_of,
    route_of_namespaced,
)
from claude_router.proxy import classify_model, route_payload

CURSOR = "composer-2.5"


def test_cursor_classification_and_namespacing() -> None:
    assert provider_of(CURSOR) == "cursor"
    assert route_of_namespaced(f"clr/cursor/{CURSOR}") == "cursor"
    assert original_model(f"clr/cursor/{CURSOR}") == CURSOR
    assert namespaced_model(CURSOR) == f"{CURSOR_MODEL_PREFIX}{CURSOR}"
    assert classify_model(f"clr/cursor/{CURSOR}", {CURSOR}) == ("cursor", CURSOR)
    with pytest.raises(ValueError, match="Cursor model is not in the clr favorites allowlist"):
        classify_model(f"clr/cursor/{CURSOR}", set())


def test_cursor_route_payload_rewrites_model() -> None:
    payload = {
        "model": f"clr/cursor/{CURSOR}",
        "stream": False,
        "messages": [{"role": "user", "content": "hello"}],
    }

    route, model, body = route_payload(json.dumps(payload).encode(), {CURSOR})
    routed = json.loads(body)

    assert (route, model, routed["model"]) == ("cursor", CURSOR, CURSOR)


def test_flatten_renders_system_and_turns() -> None:
    payload = {
        "system": "You are a research assistant.",
        "messages": [
            {"role": "user", "content": "What is foo?"},
            {"role": "assistant", "content": [{"type": "text", "text": "Foo is bar."}]},
            {"role": "user", "content": "Thanks"},
        ],
    }

    prompt = flatten_conversation(payload)

    assert "<system>\nYou are a research assistant.\n</system>" in prompt
    assert "[User]\nWhat is foo?" in prompt
    assert "[Assistant]\nFoo is bar." in prompt
    assert prompt.rstrip().endswith("no preamble or meta commentary.")


def test_images_from_last_user_message() -> None:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "aGk=",
                        },
                    },
                ],
            }
        ]
    }

    assert _images_from_last_user(payload["messages"]) == [
        {"data": "aGk=", "mimeType": "image/png"}
    ]
    assert _images_from_last_user([]) == []


def test_registry_prefix_key_and_capacity() -> None:
    registry = AgentRegistry(capacity=2)
    assert registry.prefix_key([]) is None
    assert registry.prefix_key([{"role": "user", "content": "hi"}]) is None

    history = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    key = AgentRegistry.prefix_key(history)
    assert key is not None
    registry.remember(key, "bc-1")
    assert registry.lookup(key) == "bc-1"
    assert registry.lookup("missing") is None

    other = AgentRegistry.prefix_key([*history, {"role": "assistant", "content": "d"}])
    registry.remember(other, "bc-2")
    registry.remember(
        AgentRegistry.prefix_key(
            [{"role": "user", "content": "x"}, {"role": "user", "content": "y"}]
        ),
        "bc-3",
    )
    assert registry.lookup(other) == "bc-2"
    assert registry.lookup(key) is None  # evicted, capacity 2


def test_resolve_agent_creates_on_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_request_json(method, path, credential, body=None):
        calls.append((method, path))
        if path == "/v1/agents":
            assert body["model"] == {"id": CURSOR}
            assert "transcript" in body["prompt"]["text"]
            return {"agent": {"id": "bc-agent"}, "run": {"id": "run-1"}}
        raise AssertionError("unexpected path")

    monkeypatch.setattr(cursor_module, "_request_json", fake_request_json)
    registry = AgentRegistry()
    payload = {"messages": [{"role": "user", "content": "hello"}]}

    agent_id, run_id = _resolve_agent(
        payload, CURSOR, registry, "key", None, "plan"
    )

    assert (agent_id, run_id) == ("bc-agent", "run-1")
    assert calls == [("POST", "/v1/agents")]


def test_resolve_agent_reuses_agent_on_prefix_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posts: list[str] = []

    def fake_request_json(method, path, credential, body=None):
        posts.append(path)
        if path == "/v1/agents":
            return {"agent": {"id": "bc-agent"}, "run": {"id": "run-1"}}
        assert body["prompt"]["text"] == "follow-up"
        return {"run": {"id": "run-2"}}

    monkeypatch.setattr(cursor_module, "_request_json", fake_request_json)
    registry = AgentRegistry()
    first = {"messages": [{"role": "user", "content": "hello"}]}
    _resolve_agent(first, CURSOR, registry, "key", None, "plan")
    _remember_conversation(first, registry, "bc-agent", "answer")

    follow_up = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "follow-up"},
        ]
    }
    agent_id, run_id = _resolve_agent(
        follow_up, CURSOR, registry, "key", None, "plan"
    )

    assert (agent_id, run_id) == ("bc-agent", "run-2")
    assert posts == ["/v1/agents", "/v1/agents/bc-agent/runs"]


class FakeResponse:
    def __init__(self, chunks: list[bytes]):
        self._chunks = iter(chunks)
        self.status = 200

    def read1(self, size: int) -> bytes:
        try:
            return next(self._chunks)
        except StopIteration:
            return b""


class FakeConnection:
    closed = False

    def close(self) -> None:
        self.closed = True


def test_bridge_frames_translate_run_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        (b'event: status\ndata: {"runId":"r","status":"RUNNING"}\n\n'),
        (b'event: assistant\ndata: {"text":"One"}\n\n'),
        (b'event: assistant\ndata: {"text":" two"}\n\n'),
        (
            b'event: result\ndata: {"runId":"r","status":"FINISHED",'
            b'"text":"One two"}\n\n'
        ),
        b'event: done\ndata: {}\n\n',
    ]

    def fake_resolve(payload, model, registry, credential, repos, mode):
        return "bc-agent", "run-1"

    def fake_open(agent_id, run_id, credential):
        assert (agent_id, run_id) == ("bc-agent", "run-1")
        return FakeConnection(), FakeResponse(events)

    monkeypatch.setattr(cursor_module, "_resolve_agent", fake_resolve)
    monkeypatch.setattr(cursor_module, "open_stream", fake_open)
    payload = {"messages": [{"role": "user", "content": "count"}]}

    frames = list(bridge_frames(payload, CURSOR, AgentRegistry(), "key"))
    parsed = []
    for frame in frames:
        lines = frame.decode().strip().splitlines()
        parsed.append((lines[0].removeprefix("event: "), json.loads(lines[1][6:])))

    kinds = [name for name, _ in parsed]
    assert kinds == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert parsed[0][1]["message"]["model"] == CURSOR
    assert parsed[2][1]["delta"] == {"type": "text_delta", "text": "One"}
    assert parsed[-2][1]["delta"]["stop_reason"] == "end_turn"


def test_run_messages_accumulates_text(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        b'event: assistant\ndata: {"text":"hel"}\n\n',
        b'event: assistant\ndata: {"text":"lo"}\n\n',
        b'event: result\ndata: {"status":"FINISHED","text":"hello"}\n\n',
        b'event: done\ndata: {}\n\n',
    ]
    monkeypatch.setattr(
        cursor_module,
        "_resolve_agent",
        lambda payload, model, registry, credential, repos, mode: ("bc", "run"),
    )
    monkeypatch.setattr(
        cursor_module,
        "open_stream",
        lambda agent, run, cred: (FakeConnection(), FakeResponse(events)),
    )
    payload = {"messages": [{"role": "user", "content": "hi"}]}

    response = run_messages(payload, CURSOR, AgentRegistry(), "key")

    assert response["content"] == [{"type": "text", "text": "hello"}]
    assert response["stop_reason"] == "end_turn"
    assert response["model"] == CURSOR


def test_bridge_frames_report_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [b'event: result\ndata: {"status":"FAILED"}\n\n']
    monkeypatch.setattr(
        cursor_module,
        "_resolve_agent",
        lambda payload, model, registry, credential, repos, mode: ("bc", "run"),
    )
    monkeypatch.setattr(
        cursor_module,
        "open_stream",
        lambda agent, run, cred: (FakeConnection(), FakeResponse(events)),
    )

    with pytest.raises(CursorBridgeError, match="FAILED"):
        list(bridge_frames({"messages": []}, CURSOR, AgentRegistry(), "key"))


def test_error_frame_shape() -> None:
    frame = error_frame("boom").decode()
    assert frame.startswith("event: error\n")
    assert json.loads(frame.splitlines()[1][6:])["error"]["type"] == "api_error"


def test_picker_row_labels_cursor_models() -> None:
    row = picker_row({"id": CURSOR, "name": "Composer 2.5"}, hybrid=True)

    assert row["model"] == f"clr/cursor/{CURSOR}"
    assert row["label"].endswith(" · Cursor")
    assert "Cursor Cloud Agents" in row["description"]
