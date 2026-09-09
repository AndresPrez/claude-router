"""Tests for the Cursor CLI bridge (clr/cursor/<model> route)."""

import json

import pytest

from claude_router import cursor as cursor_module
from claude_router.cursor import (
    CursorBridgeError,
    _final_result,
    _require_text_result,
    cursor_command,
    error_frame,
    flatten_conversation,
    iter_stream_events,
    run_turn,
    sse_frames,
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


def test_flatten_renders_tool_blocks_and_images() -> None:
    payload = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "grep", "input": {"q": "foo"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": "match on line 3"}],
                    },
                    {"type": "image", "source": {}},
                ],
            },
        ]
    }

    prompt = flatten_conversation(payload)

    assert "[Assistant used tool grep with:" in prompt
    assert '"q": "foo"' in prompt
    assert "[Tool result for t1] match on line 3" in prompt
    assert "[image omitted]" in prompt


def test_cursor_command_shape() -> None:
    plain = cursor_command(CURSOR, workspace="/repo", cli_path="/usr/local/bin/agent")
    assert plain[:3] == ["/usr/local/bin/agent", "--print", "--trust"]
    assert "--trust" in plain and "--mode" in plain and "ask" in plain
    assert plain[plain.index("--workspace") + 1] == "/repo"

    stream = cursor_command(CURSOR, output_format="stream-json")
    assert "stream-json" in stream
    assert "--stream-partial-output" in stream


def test_final_result_takes_last_result_line() -> None:
    lines = iter(
        [
            ("stdout", json.dumps({"type": "system", "subtype": "init"})),
            ("stdout", json.dumps({"type": "assistant", "message": {}})),
            (
                "stdout",
                json.dumps(
                    {
                        "type": "result",
                        "is_error": False,
                        "result": "hi",
                        "usage": {"outputTokens": 3},
                    }
                ),
            ),
        ]
    )

    result = _final_result(lines)

    assert result["result"] == "hi"


def test_final_result_raises_without_result() -> None:
    lines = iter([("stdout", json.dumps({"type": "system"})), ("stderr", "boom")])

    with pytest.raises(CursorBridgeError, match="no result event"):
        _final_result(lines)


def test_require_text_result_maps_usage() -> None:
    result = {
        "is_error": False,
        "result": "ok",
        "usage": {
            "inputTokens": 10,
            "outputTokens": 4,
            "cacheReadTokens": 7,
            "cacheWriteTokens": 1,
        },
    }

    text, usage = _require_text_result(CURSOR, result)

    assert text == "ok"
    assert usage == {
        "input_tokens": 10,
        "output_tokens": 4,
        "cache_read_input_tokens": 7,
        "cache_creation_input_tokens": 1,
    }


def test_require_text_result_rejects_error_results() -> None:
    with pytest.raises(CursorBridgeError, match="reported failure"):
        _require_text_result(CURSOR, {"is_error": True, "result": "quota exceeded"})
    with pytest.raises(CursorBridgeError, match="did not include assistant text"):
        _require_text_result(CURSOR, {"is_error": False})


def test_run_turn_builds_messages_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cursor_module.shutil, "which", lambda _: "/usr/local/bin/agent")

    class Completed:
        stdout = json.dumps(
            {
                "type": "result",
                "is_error": False,
                "result": "BRIDGE_OK",
                "usage": {"inputTokens": 6909, "outputTokens": 56},
            }
        )
        stderr = ""

    seen: dict[str, object] = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["input"] = kwargs.get("input")
        return Completed()

    monkeypatch.setattr(cursor_module.subprocess, "run", fake_run)
    response = run_turn(CURSOR, "say ok", "/repo")

    assert response["model"] == CURSOR
    assert response["content"] == [{"type": "text", "text": "BRIDGE_OK"}]
    assert response["stop_reason"] == "end_turn"
    assert response["usage"]["input_tokens"] == 6909
    assert seen["input"] == "say ok"
    assert seen["command"][list(seen["command"]).index("--model") + 1] == CURSOR  # type: ignore[index]


def test_run_turn_requires_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cursor_module, "CURSOR_CLI_CANDIDATES", ("/nonexistent/agent",))
    monkeypatch.setattr(cursor_module.shutil, "which", lambda _: None)

    with pytest.raises(CursorBridgeError, match="not found"):
        run_turn(CURSOR, "hi")


def _parse_frames(frames: list[bytes]) -> list[tuple[str, dict]]:
    parsed = []
    for frame in frames:
        lines = frame.decode().strip().splitlines()
        event = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        parsed.append((event, data))
    return parsed


def test_sse_frames_synthesize_anthropic_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    events = iter(
        [
            {"type": "system", "subtype": "init"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "One"}]}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": " two"}]}},
            {
                "type": "result",
                "is_error": False,
                "result": "One two",
                "usage": {"inputTokens": 5, "outputTokens": 2},
            },
        ]
    )
    monkeypatch.setattr(cursor_module, "iter_stream_events", lambda *a, **k: events)

    frames = _parse_frames(list(sse_frames(CURSOR, "count")))

    kinds = [name for name, _ in frames]
    assert kinds == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert frames[0][1]["message"]["model"] == CURSOR
    assert frames[2][1]["delta"] == {"type": "text_delta", "text": "One"}
    assert frames[-2][1]["usage"]["output_tokens"] == 2
    assert frames[-2][1]["delta"]["stop_reason"] == "end_turn"


def test_sse_frames_error_event() -> None:
    frame = error_frame("cursor agent timed out")
    parsed = _parse_frames([frame])[0]

    assert parsed[0] == "error"
    assert parsed[1]["error"]["type"] == "api_error"


def test_iter_stream_events_parses_and_stops_at_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = "\n".join(
        [
            json.dumps({"type": "system", "subtype": "init"}),
            json.dumps(
                {"type": "assistant", "message": {"content": [{"type": "text", "text": "x"}]}}
            ),
            json.dumps({"type": "result", "is_error": False, "result": "x"}),
            json.dumps({"type": "result", "is_error": False, "result": "never"}),
        ]
    )

    class FakeStdin:
        def __init__(self):
            self.written = []
            self.closed = False

        def write(self, data):
            self.written.append(data)

        def close(self):
            self.closed = True

    class FakeProcess:
        def __init__(self):
            self.stdin = FakeStdin()
            self.stdout = iter(payload.splitlines())
            self.returncode = 0

        def poll(self):
            return 0

        def kill(self):
            pass

        def wait(self):
            return 0

    monkeypatch.setattr(cursor_module.shutil, "which", lambda _: "/usr/local/bin/agent")
    monkeypatch.setattr(cursor_module.subprocess, "Popen", lambda *a, **k: FakeProcess())

    events = list(iter_stream_events(CURSOR, "hi"))

    assert [event["type"] for event in events] == ["system", "assistant", "result"]


def test_picker_row_labels_cursor_models() -> None:
    row = picker_row({"id": CURSOR, "name": "Composer 2.5"}, hybrid=True)

    assert row["model"] == f"clr/cursor/{CURSOR}"
    assert row["label"].endswith(" · Cursor")
    assert "Cursor team plan" in row["description"]
