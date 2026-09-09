"""Cursor CLI bridge: serve ``clr/cursor/<model>`` requests via the ``agent`` CLI.

Cursor's team plan exposes models only through the local Cursor Agent CLI
(``agent``), which runs its own tool loop rather than speaking a raw wire
protocol. This bridge flattens an Anthropic Messages request into a single
prompt, runs one read-only (``--mode ask``) CLI turn, and maps the outcome
back into Anthropic response shapes (JSON or synthesized SSE). Tool
definitions are rendered as transcript context only; the bridge never
fabricates ``tool_use`` blocks.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

CURSOR_CLI = "agent"
# launchd services get a minimal PATH, so probe the usual install locations
# before falling back to PATH lookup.
CURSOR_CLI_CANDIDATES = ("~/.local/bin/agent", "/usr/local/bin/agent")
CURSOR_TIMEOUT_SECONDS = 600
_MAX_DIAGNOSTIC_CHARS = 400


def resolve_cli_path() -> str | None:
    """Return the Cursor CLI path, probing known locations then PATH."""
    for candidate in CURSOR_CLI_CANDIDATES:
        path = Path(candidate).expanduser()
        if path.is_file():
            return str(path)
    return shutil.which(CURSOR_CLI)


class CursorBridgeError(RuntimeError):
    """Raised when the Cursor CLI bridge cannot complete a turn."""


def _blocks_to_text(content: Any) -> str:
    """Render one message content (string or block list) as transcript text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
        elif block_type == "tool_use":
            name = block.get("name")
            payload = json.dumps(block.get("input"), ensure_ascii=False)
            parts.append(f"[Assistant used tool {name} with: {payload}]")
        elif block_type == "tool_result":
            inner = _blocks_to_text(block.get("content"))
            tool_use_id = block.get("tool_use_id")
            parts.append(f"[Tool result for {tool_use_id}] {inner}".rstrip())
        elif block_type == "image":
            parts.append("[image omitted]")
        else:
            payload = json.dumps(block, ensure_ascii=False)
            parts.append(f"[unsupported block: {payload[:200]}]")
    return "\n\n".join(part for part in parts if part)


def flatten_conversation(payload: dict[str, Any]) -> str:
    """Flatten an Anthropic Messages payload into one Cursor CLI prompt."""
    sections: list[str] = []

    system = _blocks_to_text(payload.get("system"))
    if system:
        sections.append(f"<system>\n{system}\n</system>")

    turns: list[str] = []
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = "User" if message.get("role") == "user" else "Assistant"
            text = _blocks_to_text(message.get("content"))
            if text:
                turns.append(f"[{role}]\n{text}")
    if turns:
        sections.append("<transcript>\n" + "\n\n".join(turns) + "\n</transcript>")

    sections.append(
        "Continue the transcript above: reply with only the next Assistant "
        "message, with no preamble or meta commentary."
    )
    return "\n\n".join(sections)


def cursor_command(
    model_id: str,
    *,
    output_format: str = "json",
    workspace: str | None = None,
    cli_path: str | None = None,
) -> list[str]:
    """Build the read-only, non-interactive Cursor CLI invocation."""
    command = [
        cli_path or resolve_cli_path() or CURSOR_CLI,
        "--print",
        "--trust",
        "--mode",
        "ask",
        "--model",
        model_id,
        "--output-format",
        output_format,
    ]
    if output_format == "stream-json":
        command.append("--stream-partial-output")
    if workspace:
        command.extend(["--workspace", workspace])
    return command


def _usage_from(result: dict[str, Any]) -> dict[str, int]:
    usage = result.get("usage")
    if not isinstance(usage, dict):
        usage = {}

    def count(key: str) -> int:
        value = usage.get(key)
        return value if isinstance(value, int) and value >= 0 else 0

    return {
        "input_tokens": count("inputTokens"),
        "output_tokens": count("outputTokens"),
        "cache_read_input_tokens": count("cacheReadTokens"),
        "cache_creation_input_tokens": count("cacheWriteTokens"),
    }


def _final_result(lines: Iterator[tuple[str, str]]) -> dict[str, Any]:
    """Return the last ``type == "result"`` JSON line, or raise with context."""
    result: dict[str, Any] | None = None
    tail = ""
    for stream, line in lines:
        if stream == "stderr":
            tail = line[-_MAX_DIAGNOSTIC_CHARS:]
            continue
        tail = line[-_MAX_DIAGNOSTIC_CHARS:]
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            result = event
    if result is None:
        raise CursorBridgeError(
            f"cursor CLI produced no result event; output tail: {tail or '(empty)'}"
        )
    return result


def _message_id() -> str:
    return f"msg_cursor_{uuid.uuid4().hex[:12]}"


def _require_text_result(model_id: str, result: dict[str, Any]) -> tuple[str, dict[str, int]]:
    if result.get("is_error") is True:
        detail = str(result.get("result") or result.get("subtype") or "unknown error")
        raise CursorBridgeError(f"cursor agent reported failure: {detail[:_MAX_DIAGNOSTIC_CHARS]}")
    text = result.get("result")
    if not isinstance(text, str):
        raise CursorBridgeError("cursor CLI result did not include assistant text")
    return text, _usage_from(result)


def run_turn(
    model_id: str,
    prompt: str,
    workspace: str | None = None,
    *,
    timeout: int = CURSOR_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run one Cursor CLI turn and return an Anthropic Messages response."""
    cli_path = resolve_cli_path()
    if cli_path is None:
        raise CursorBridgeError(f"Cursor CLI '{CURSOR_CLI}' not found")
    command = cursor_command(model_id, workspace=workspace, cli_path=cli_path)
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            command,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise CursorBridgeError(f"cursor agent timed out after {timeout}s") from exc
    lines = [("stdout", line) for line in completed.stdout.splitlines()]
    lines += [("stderr", line) for line in completed.stderr.splitlines()]
    result = _final_result(iter(lines))
    text, usage = _require_text_result(model_id, result)
    return {
        "id": _message_id(),
        "type": "message",
        "role": "assistant",
        "model": model_id,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": usage,
    }


def iter_stream_events(
    model_id: str,
    prompt: str,
    workspace: str | None = None,
    *,
    timeout: int = CURSOR_TIMEOUT_SECONDS,
) -> Iterator[dict[str, Any]]:
    """Yield parsed ``stream-json`` events from one Cursor CLI turn."""
    cli_path = resolve_cli_path()
    if cli_path is None:
        raise CursorBridgeError(f"Cursor CLI '{CURSOR_CLI}' not found")
    command = cursor_command(
        model_id, output_format="stream-json", workspace=workspace, cli_path=cli_path
    )
    process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert process.stdin is not None and process.stdout is not None
    deadline = time.monotonic() + timeout
    try:
        process.stdin.write(prompt)
        process.stdin.close()
    except (BrokenPipeError, OSError) as exc:
        raise CursorBridgeError(f"cursor CLI rejected the prompt: {exc}") from exc
    try:
        for line in process.stdout:
            if time.monotonic() > deadline:
                raise CursorBridgeError(f"cursor agent timed out after {timeout}s")
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                yield event
            if event.get("type") == "result":
                break
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def _sse_frame(event: str, data: dict[str, Any]) -> bytes:
    body = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    return f"event: {event}\ndata: {body}\n\n".encode()


def error_frame(message: str) -> bytes:
    """Build an Anthropic-style SSE error event for mid-stream failures."""
    return _sse_frame(
        "error", {"type": "error", "error": {"type": "api_error", "message": message}}
    )


def sse_frames(
    model_id: str,
    prompt: str,
    workspace: str | None = None,
    *,
    timeout: int = CURSOR_TIMEOUT_SECONDS,
) -> Iterator[bytes]:
    """Run one Cursor turn and yield Anthropic SSE frames for it."""
    started = False
    text_open = False
    usage: dict[str, int] = {}
    result_seen = False
    events = iter_stream_events(model_id, prompt, workspace, timeout=timeout)
    while True:
        try:
            event = next(events)
        except StopIteration:
            break
        kind = event.get("type")
        if not started:
            started = True
            yield _sse_frame(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": _message_id(),
                        "type": "message",
                        "role": "assistant",
                        "model": model_id,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        },
                    },
                },
            )
        if kind == "assistant":
            content = event.get("message")
            blocks = content.get("content") if isinstance(content, dict) else None
            for block in blocks if isinstance(blocks, list) else []:
                if not isinstance(block, dict) or block.get("type") != "text":
                    continue
                delta = block.get("text")
                if not isinstance(delta, str) or not delta:
                    continue
                if not text_open:
                    text_open = True
                    yield _sse_frame(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                yield _sse_frame(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": delta},
                    },
                )
        elif kind == "result":
            result_seen = True
            _, usage = _require_text_result(model_id, event)
    if not started:
        raise CursorBridgeError("cursor CLI produced no events")
    if not result_seen:
        raise CursorBridgeError("cursor CLI stream ended without a result event")
    if text_open:
        yield _sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield _sse_frame(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "cache_read_input_tokens": usage["cache_read_input_tokens"],
                "cache_creation_input_tokens": usage["cache_creation_input_tokens"],
            },
        },
    )
    yield _sse_frame("message_stop", {"type": "message_stop"})
