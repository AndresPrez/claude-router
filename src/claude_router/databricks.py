"""Databricks AI Gateway bridge: ``clr/databricks/<model>`` via the Responses API.

Databricks model serving exposes an OpenAI Responses-compatible endpoint at
``{base}/ai-gateway/mlflow/v1/responses`` (per-deployment base URL, Bearer
token). This bridge translates Anthropic Messages requests to Responses
shape and back - non-streaming JSON and SSE - so Claude Code can drive
gateway models like ``system.ai.glm-5-3-flash``.

Translation map (Anthropic -> Responses):
  system                -> instructions
  messages[] text       -> input[] {role, content:[{type: input_text|output_text}]}
  assistant tool_use    -> input item {type: function_call, call_id, name, arguments}
  user tool_result      -> input item {type: function_call_output, call_id, output}
  tools (input_schema)  -> tools [{type: function, parameters}]
  max_tokens            -> max_output_tokens
  thinking/effort       -> reasoning {effort}
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlsplit

from .paths import databricks_credential_path
from .storage import atomic_write_text


class DatabricksBridgeError(RuntimeError):
    """Raised when the Databricks gateway exchange fails."""


def validate_databricks_key_shape(key: str) -> None:
    if len(key) < 20 or any(character.isspace() for character in key):
        raise ValueError(
            "the Databricks token has an unexpected format (a token of 20+ characters)"
        )


def read_databricks_credential() -> str:
    path = databricks_credential_path()
    try:
        key = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Databricks token not found at {path}; run `clr config --databricks-key`"
        ) from exc
    validate_databricks_key_shape(key)
    return key


def write_databricks_credential(key: str) -> None:
    validate_databricks_key_shape(key)
    atomic_write_text(databricks_credential_path(), f"{key}\n", 0o600)


def _content_text(blocks: Any) -> str:
    if isinstance(blocks, str):
        return blocks
    if not isinstance(blocks, list):
        return ""
    parts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
    return "\n\n".join(p for p in parts if p)


def translate_request(payload: dict[str, Any], reasoning_effort: str | None) -> dict[str, Any]:
    """Map an Anthropic Messages payload to a Responses API body."""
    body: dict[str, Any] = {"model": payload["model"]}

    system = _content_text(payload.get("system"))
    if system:
        body["instructions"] = system

    items: list[dict[str, Any]] = []
    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = "assistant" if message.get("role") == "assistant" else "user"
        content = message.get("content")
        texts: list[str] = []
        if isinstance(content, str):
            texts = [content]
            content = []
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind in ("text", "thinking"):
                if kind == "text":
                    texts.append(block.get("text", ""))
            elif kind == "tool_use":
                if texts:
                    items.append({
                        "role": role,
                        "content": [{"type": "output_text" if role == "assistant" else "input_text",
                                     "text": "\n\n".join(t for t in texts if t)}],
                    })
                    texts = []
                items.append({
                    "type": "function_call",
                    "call_id": block.get("id"),
                    "name": block.get("name"),
                    "arguments": json.dumps(block.get("input") or {}),
                })
            elif kind == "tool_result":
                items.append({
                    "type": "function_call_output",
                    "call_id": block.get("tool_use_id"),
                    "output": _content_text(block.get("content")) or str(block.get("content", "")),
                })
            elif kind == "image":
                texts.append("[image omitted]")
        if texts:
            items.append({
                "role": role,
                "content": [{"type": "output_text" if role == "assistant" else "input_text",
                             "text": "\n\n".join(t for t in texts if t)}],
            })
    body["input"] = items

    tools = []
    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict) or tool.get("type") not in (None, "custom"):
            continue
        tools.append({
            "type": "function",
            "name": tool.get("name"),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
        })
    if tools:
        body["tools"] = tools
        choice = payload.get("tool_choice")
        if choice == "any" or choice == {"type": "any"}:
            body["tool_choice"] = "required"
        elif isinstance(choice, dict) and choice.get("type") == "tool":
            body["tool_choice"] = {"type": "function", "name": choice.get("name")}
        elif choice in ("auto", "none"):
            body["tool_choice"] = choice

    if isinstance(payload.get("max_tokens"), int):
        body["max_output_tokens"] = payload["max_tokens"]
    for field in ("temperature", "top_p"):
        if payload.get(field) is not None:
            body[field] = payload[field]
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    if payload.get("stream"):
        body["stream"] = True
    return body


def _sse_frame(event: str, data: dict[str, Any]) -> bytes:
    body = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    return f"event: {event}\ndata: {body}\n\n".encode()


def error_frame(message: str) -> bytes:
    return _sse_frame("error", {"type": "error", "error": {"type": "api_error", "message": message}})


def translate_response(document: dict[str, Any], model: str) -> dict[str, Any]:
    """Map a non-streaming Responses result to an Anthropic Messages response."""
    content: list[dict[str, Any]] = []
    tool_calls = 0
    for item in document.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            text = "".join(
                block.get("text", "")
                for block in item.get("content") or []
                if isinstance(block, dict) and block.get("type") == "output_text"
            )
            if text:
                content.append({"type": "text", "text": text})
        elif item.get("type") == "function_call":
            tool_calls += 1
            try:
                arguments = json.loads(item.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {"_raw": item.get("arguments")}
            content.append({
                "type": "tool_use",
                "id": item.get("call_id") or item.get("id") or f"toolu_{tool_calls:03d}",
                "name": item.get("name"),
                "input": arguments,
            })
    if not content:
        content = [{"type": "text", "text": ""}]
    usage = document.get("usage") or {}
    return {
        "id": str(document.get("id") or "msg_databricks"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": "tool_use" if tool_calls else "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("input_tokens", 0) or 0,
            "output_tokens": usage.get("output_tokens", 0) or 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }


def translate_stream_events(
    upstream_events: Iterator[dict[str, Any]], model: str
) -> Iterator[bytes]:
    """Yield Anthropic SSE frames from Responses API stream events."""
    started = False
    open_block = -1
    usage: dict[str, int] = {}
    completed = False

    def ensure_started() -> bytes:
        nonlocal started
        if started:
            return b""
        started = True
        return _sse_frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_databricks_stream",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 0, "output_tokens": 0,
                        "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
                    },
                },
            },
        )

    for event in upstream_events:
        kind = event.get("type", "")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else event
        if kind.startswith("response.created"):
            yield ensure_started()
        elif kind == "response.output_item.added":
            yield ensure_started()
            item = payload.get("item") or {}
            if item.get("type") == "message":
                open_block += 1
                yield _sse_frame(
                    "content_block_start",
                    {"type": "content_block_start", "index": open_block,
                     "content_block": {"type": "text", "text": ""}},
                )
            elif item.get("type") == "function_call":
                if open_block >= 0:
                    yield _sse_frame("content_block_stop",
                                     {"type": "content_block_stop", "index": open_block})
                open_block += 1
                yield _sse_frame(
                    "content_block_start",
                    {"type": "content_block_start", "index": open_block,
                     "content_block": {"type": "tool_use", "id": item.get("call_id") or "toolu_1",
                                       "name": item.get("name", ""), "input": {}}},
                )
        elif kind == "response.output_text.delta":
            delta = payload.get("delta") or ""
            if delta:
                yield _sse_frame(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": max(open_block, 0),
                     "delta": {"type": "text_delta", "text": delta}},
                )
        elif kind == "response.function_call_arguments.delta":
            delta = payload.get("delta") or ""
            if delta:
                yield _sse_frame(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": max(open_block, 0),
                     "delta": {"type": "input_json_delta", "partial_json": delta}},
                )
        elif kind == "response.output_item.done":
            if open_block >= 0:
                yield _sse_frame("content_block_stop",
                                 {"type": "content_block_stop", "index": open_block})
                open_block = -1
        elif kind == "response.completed":
            completed = True
            response = payload.get("response") or {}
            usage = response.get("usage") or usage
            yield _sse_frame(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {
                        "input_tokens": usage.get("input_tokens", 0) or 0,
                        "output_tokens": usage.get("output_tokens", 0) or 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            )
            yield _sse_frame("message_stop", {"type": "message_stop"})
        elif kind in ("response.failed", "error", "response.incomplete"):
            detail = payload.get("error") or payload.get("response", {}).get("error") or {}
            raise DatabricksBridgeError(
                f"databricks stream failed: {detail.get('message') or kind}"
            )
    if not started:
        raise DatabricksBridgeError("databricks stream produced no events")
    if not completed:
        raise DatabricksBridgeError("databricks stream ended without completion")


def validate_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise DatabricksBridgeError(f"invalid Databricks base URL: {base_url}")
    return base_url.rstrip("/")


def responses_path(base_url: str) -> str:
    return f"{validate_base_url(base_url)}/ai-gateway/mlflow/v1/responses"


def exchange(
    base_url: str,
    credential: str,
    body: dict[str, Any],
) -> tuple[Any, Any]:
    """POST the Responses body; return (connection, response) for streaming."""
    import http.client
    import ssl
    from urllib.parse import urlsplit

    target = urlsplit(responses_path(base_url))
    conn = http.client.HTTPSConnection(
        target.hostname, target.port or 443, timeout=600, context=ssl.create_default_context()
    )
    try:
        conn.request(
            "POST",
            target.path,
            body=json.dumps(body, ensure_ascii=False).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {credential}",
                "Accept": "text/event-stream" if body.get("stream") else "application/json",
            },
        )
        response = conn.getresponse()
    except (OSError, http.client.HTTPException) as exc:
        conn.close()
        raise DatabricksBridgeError(f"databricks gateway unreachable: {exc}") from exc
    if response.status >= 400:
        detail = response.read(2048).decode(errors="replace")
        conn.close()
        raise DatabricksBridgeError(
            f"databricks gateway HTTP {response.status}: {detail}"
        )
    return conn, response


def iter_responses_events(conn: Any, response: Any) -> Iterator[dict[str, Any]]:
    """Parse the Responses SSE stream into event dicts."""
    buffer = b""
    try:
        while chunk := response.read1(65536):
            buffer += chunk
            while b"\n\n" in buffer:
                raw, buffer = buffer.split(b"\n\n", 1)
                event_name, data = "", None
                for line in raw.decode("utf-8", errors="replace").splitlines():
                    if line.startswith("event:"):
                        event_name = line[6:].strip()
                    elif line.startswith("data:"):
                        try:
                            data = json.loads(line[5:].strip())
                        except json.JSONDecodeError:
                            data = None
                if isinstance(data, dict):
                    yield {"type": data.get("type") or event_name, "payload": data}
                elif event_name:
                    yield {"type": event_name, "payload": {}}
    finally:
        conn.close()


def serve(
    payload: dict[str, Any],
    base_url: str,
    credential: str,
    reasoning_effort: str | None,
) -> Iterator[bytes] | dict[str, Any]:
    """Translate, exchange, and translate back.

    Returns a dict for non-streaming requests, else an iterator of Anthropic
    SSE frames.
    """
    body = translate_request(payload, reasoning_effort)
    conn, response = exchange(base_url, credential, body)
    if not body.get("stream"):
        try:
            document = json.loads(response.read().decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DatabricksBridgeError("databricks gateway returned invalid JSON") from exc
        finally:
            conn.close()
        return translate_response(document, str(payload.get("model", "")))
    return translate_stream_events(
        iter_responses_events(conn, response), str(payload.get("model", ""))
    )
