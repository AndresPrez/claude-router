"""Cursor Cloud Agents bridge: serve ``clr/cursor/<model>`` via api.cursor.com.

Cursor's team plan exposes models through the Cloud Agents API (public beta):
agents are durable, keep conversation and workspace state across runs, and
stream assistant text deltas over SSE. This module maps Anthropic Messages
requests onto that surface:

- the first request of a conversation creates a no-repo agent whose initial
  run carries the flattened history;
- follow-up requests whose message prefix matches a known conversation are
  sent as follow-up runs carrying only the newest user turn;
- run SSE (``assistant`` deltas, ``result``) is translated into Anthropic SSE
  frames, and a client disconnect best-effort cancels the run.

Tool definitions are not forwarded: the cloud agent runs its own tool loop,
so catalog entries advertise no Messages-API tool support.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import ssl
import threading
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlsplit

from .paths import cursor_credential_path
from .storage import atomic_write_text

CURSOR_UPSTREAM = "https://api.cursor.com"
CURSOR_STREAM_PATH = "/v1/agents/{agent_id}/runs/{run_id}/stream"
CURSOR_RUNS_PATH = "/v1/agents/{agent_id}/runs"
CURSOR_AGENTS_PATH = "/v1/agents"
_REGISTRY_CAPACITY = 256


class CursorBridgeError(RuntimeError):
    """Raised when the Cursor Cloud Agents bridge cannot complete a request."""


def validate_cursor_key_shape(key: str) -> None:
    if len(key) < 20 or any(character.isspace() for character in key):
        raise ValueError(
            "the Cursor API key has an unexpected format (a key of 20+ characters)"
        )


def read_cursor_credential() -> str:
    path = cursor_credential_path()
    try:
        key = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Cursor API key not found at {path}; generate one at "
            "cursor.com/dashboard -> API Keys and run `clr config --cursor-key`"
        ) from exc
    validate_cursor_key_shape(key)
    return key


def write_cursor_credential(key: str) -> None:
    validate_cursor_key_shape(key)
    atomic_write_text(cursor_credential_path(), f"{key}\n", 0o600)


class AgentRegistry:
    """Map conversation prefixes to durable cloud agent ids (thread-safe LRU)."""

    def __init__(self, capacity: int = _REGISTRY_CAPACITY) -> None:
        self.capacity = capacity
        self._entries: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _history_key(messages: list[Any]) -> str:
        blob = json.dumps(messages, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    @classmethod
    def prefix_key(cls, messages: list[Any]) -> str | None:
        """Hash all messages except the newest; ``None`` for first requests."""
        if not isinstance(messages, list) or len(messages) < 2:
            return None
        return cls._history_key(messages[:-1])

    def lookup(self, key: str | None) -> str | None:
        if key is None:
            return None
        with self._lock:
            agent_id = self._entries.get(key)
            if agent_id is not None:
                self._entries.move_to_end(key)
            return agent_id

    def remember(self, key: str | None, agent_id: str) -> None:
        if key is None:
            return
        with self._lock:
            self._entries[key] = agent_id
            self._entries.move_to_end(key)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)


def _blocks_to_text(content: Any) -> str:
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
            parts.append(f"[Tool result for {block.get('tool_use_id')}] {inner}".rstrip())
        elif block_type == "image":
            parts.append("[image attached separately]")
        else:
            payload = json.dumps(block, ensure_ascii=False)
            parts.append(f"[unsupported block: {payload[:200]}]")
    return "\n\n".join(part for part in parts if part)


def _images_from_last_user(messages: list[Any]) -> list[dict[str, str]]:
    """Extract base64 images from the newest user message, newest-first order."""
    for message in reversed(messages if isinstance(messages, list) else []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        images: list[dict[str, str]] = []
        content = message.get("content")
        if not isinstance(content, list):
            return images
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "image":
                continue
            source = block.get("source")
            if (
                isinstance(source, dict)
                and source.get("type") == "base64"
                and isinstance(source.get("data"), str)
            ):
                mime = source.get("media_type")
                fallback = "image/png"
                images.append(
                    {
                        "data": source["data"],
                        "mimeType": mime if isinstance(mime, str) else fallback,
                    }
                )
                if len(images) == 5:
                    return images
        return images
    return []


def flatten_conversation(payload: dict[str, Any]) -> str:
    """Flatten an Anthropic Messages payload into one Cloud Agent prompt."""
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


def _open(upstream: str, timeout: int) -> http.client.HTTPSConnection:
    parsed = urlsplit(upstream)
    if parsed.scheme != "https" or not parsed.hostname:
        raise CursorBridgeError(f"invalid Cursor upstream: {upstream}")
    return http.client.HTTPSConnection(
        parsed.hostname, parsed.port or 443, timeout=timeout, context=ssl.create_default_context()
    )


def _request_json(
    method: str, path: str, credential: str, body: dict[str, Any] | None = None
) -> dict[str, Any]:
    connection = _open(CURSOR_UPSTREAM, timeout=120)
    headers = {
        "Authorization": f"Bearer {credential}",
        "Accept": "application/json",
    }
    payload = None
    if body is not None:
        payload = json.dumps(body, ensure_ascii=False).encode()
        headers["Content-Type"] = "application/json"
    try:
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        if response.status >= 400:
            detail = raw[:2048].decode(errors="replace")
            raise CursorBridgeError(
                f"cursor API {method} {path} failed: HTTP {response.status} {detail}"
            )
    except CursorBridgeError:
        raise
    except (OSError, http.client.HTTPException) as exc:
        raise CursorBridgeError(f"cursor API request failed: {exc}") from exc
    finally:
        connection.close()
    try:
        document = json.loads(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CursorBridgeError(f"cursor API returned invalid JSON for {path}") from exc
    if not isinstance(document, dict):
        raise CursorBridgeError(f"cursor API returned a non-object for {path}")
    return document


def open_stream(
    agent_id: str, run_id: str, credential: str
) -> tuple[http.client.HTTPSConnection, http.client.HTTPSResponse]:
    """Open the run SSE stream; caller owns the returned connection."""
    path = CURSOR_STREAM_PATH.format(agent_id=agent_id, run_id=run_id)
    connection = _open(CURSOR_UPSTREAM, timeout=600)
    try:
        connection.request(
            "GET",
            path,
            headers={
                "Authorization": f"Bearer {credential}",
                "Accept": "text/event-stream",
            },
        )
        response = connection.getresponse()
    except (OSError, http.client.HTTPException) as exc:
        connection.close()
        raise CursorBridgeError(f"cursor run stream failed: {exc}") from exc
    if response.status >= 400:
        detail = response.read(2048).decode(errors="replace")
        connection.close()
        raise CursorBridgeError(
            f"cursor run stream failed: HTTP {response.status} {detail}"
        )
    return connection, response


def _resolve_agent(
    payload: dict[str, Any],
    model: str,
    registry: AgentRegistry,
    credential: str,
    repos: list[str] | None,
    mode: str,
) -> tuple[str, str]:
    """Return ``(agent_id, run_id)``, creating the agent on a prefix miss."""
    messages = payload.get("messages")
    key = AgentRegistry.prefix_key(messages) if isinstance(messages, list) else None
    agent_id = registry.lookup(key)
    if agent_id is not None:
        last = payload.get("messages", [])
        text = _blocks_to_text(last[-1].get("content") if isinstance(last, list) else "")
        document = _request_json(
            "POST",
            CURSOR_RUNS_PATH.format(agent_id=agent_id),
            credential,
            {"prompt": {"text": text}},
        )
        run = document.get("run")
        if not isinstance(run, dict) or not isinstance(run.get("id"), str):
            raise CursorBridgeError("cursor API run response missing run.id")
        return agent_id, str(run["id"])

    images = _images_from_last_user(messages) if isinstance(messages, list) else []
    body: dict[str, Any] = {
        "prompt": {"text": flatten_conversation(payload)},
        "model": {"id": model},
        "mode": mode,
    }
    if images:
        body["prompt"]["images"] = images
    if repos:
        body["repos"] = [{"url": url} for url in repos]
    document = _request_json("POST", CURSOR_AGENTS_PATH, credential, body)
    agent = document.get("agent")
    run = document.get("run")
    if not isinstance(agent, dict) or not isinstance(agent.get("id"), str):
        raise CursorBridgeError("cursor API agent response missing agent.id")
    if not isinstance(run, dict) or not isinstance(run.get("id"), str):
        raise CursorBridgeError("cursor API agent response missing run.id")
    if key is not None:
        registry.remember(key, str(agent["id"]))
    return str(agent["id"]), str(run["id"])


def _remember_conversation(
    payload: dict[str, Any], registry: AgentRegistry, agent_id: str, reply: str
) -> None:
    """Bind the next request's prefix (this history + reply) to the agent."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return
    extended = [*messages, {"role": "assistant", "content": reply}]
    registry.remember(AgentRegistry._history_key(extended), agent_id)


def _upstream_events(
    connection: http.client.HTTPSConnection, response: http.client.HTTPSResponse
) -> Iterator[dict[str, Any]]:
    """Yield parsed SSE events from the run stream until ``done`` or EOF."""
    buffer = b""
    event_name = ""
    data_lines: list[str] = []
    try:
        while True:
            chunk = response.read1(64 * 1024)
            if not chunk:
                break
            buffer += chunk
            while True:
                match = _next_event(buffer)
                if match is None:
                    break
                raw, buffer = match
                for line in raw.decode("utf-8", errors="replace").splitlines():
                    if line.startswith("event:"):
                        event_name = line[6:].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].strip())
                if not event_name and not data_lines:
                    continue
                try:
                    document = json.loads("\n".join(data_lines)) if data_lines else {}
                except json.JSONDecodeError:
                    document = {}
                yield {"event": event_name, "data": document}
                event_name = ""
                data_lines = []
                if document.get("type") == "message_stop" or event_name == "done":
                    return
    finally:
        connection.close()


def _next_event(buffer: bytes) -> tuple[bytes, bytes] | None:
    endings = [
        (position, separator)
        for separator in (b"\n\n", b"\r\n\r\n")
        if (position := buffer.find(separator)) >= 0
    ]
    if not endings:
        return None
    position, separator = min(endings, key=lambda item: item[0])
    end = position + len(separator)
    return buffer[:end], buffer[end:]


def _sse_frame(event: str, data: dict[str, Any]) -> bytes:
    body = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    return f"event: {event}\ndata: {body}\n\n".encode()


def error_frame(message: str) -> bytes:
    """Build an Anthropic-style SSE error event for mid-stream failures."""
    return _sse_frame(
        "error", {"type": "error", "error": {"type": "api_error", "message": message}}
    )


def _message_id() -> str:
    return f"msg_cursor_{uuid.uuid4().hex[:12]}"


def bridge_frames(
    payload: dict[str, Any],
    model: str,
    registry: AgentRegistry,
    credential: str,
    *,
    repos: list[str] | None = None,
    mode: str = "plan",
    active: dict[str, str] | None = None,
) -> Iterator[bytes]:
    """Run one Cursor turn and yield Anthropic SSE frames for it.

    When ``active`` is provided it is filled with ``agent_id``/``run_id``/
    ``credential`` as soon as a run exists, so callers can cancel it if the
    client disconnects mid-stream.
    """
    agent_id, run_id = _resolve_agent(
        payload, model, registry, credential, repos, mode
    )
    if active is not None:
        active.update(
            {"agent_id": agent_id, "run_id": run_id, "credential": credential}
        )
    connection, response = open_stream(agent_id, run_id, credential)

    started = False
    text_open = False
    final_text = ""
    reply_text = ""
    try:
        for event in _upstream_events(connection, response):
            kind = event["event"]
            data = event["data"]
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
                            "model": model,
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
                delta = data.get("text")
                if isinstance(delta, str) and delta:
                    reply_text += delta
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
                status = data.get("status")
                text = data.get("text")
                if isinstance(text, str) and text:
                    final_text = text
                if status != "FINISHED":
                    raise CursorBridgeError(f"cursor run ended with status {status}")
            elif kind == "error":
                raise CursorBridgeError(
                    f"cursor run stream error: {data.get('code')} {data.get('message')}"
                )
    except CursorBridgeError:
        raise
    if not started:
        raise CursorBridgeError("cursor run stream produced no events")
    if not text_open and final_text:
        reply_text = final_text
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
                "delta": {"type": "text_delta", "text": reply_text},
            },
        )
    if text_open:
        yield _sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield _sse_frame(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    )
    yield _sse_frame("message_stop", {"type": "message_stop"})
    _remember_conversation(payload, registry, agent_id, reply_text or final_text)


def run_messages(
    payload: dict[str, Any],
    model: str,
    registry: AgentRegistry,
    credential: str,
    *,
    repos: list[str] | None = None,
    mode: str = "plan",
    active: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one Cursor turn and return a complete Anthropic Messages response."""
    text = ""
    for frame in bridge_frames(
        payload, model, registry, credential, repos=repos, mode=mode, active=active
    ):
        event = frame.decode()
        for line in event.splitlines():
            if line.startswith("data:"):
                try:
                    document = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                delta = (
                    document.get("delta", {}).get("text")
                    if isinstance(document.get("delta"), dict)
                    else None
                )
                if isinstance(delta, str):
                    text += delta
                block = document.get("content_block")
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    text += block["text"]
    return {
        "id": _message_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }


def cancel_run(agent_id: str, run_id: str, credential: str) -> None:
    """Best-effort cancellation of an active run (ignored on failure)."""
    try:
        _request_json(
            "POST",
            f"/v1/agents/{agent_id}/runs/{run_id}/cancel",
            credential,
            {},
        )
    except CursorBridgeError:
        return
    except KeyError:
        return
