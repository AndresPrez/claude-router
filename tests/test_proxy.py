from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from claude_router.models import ZAI_MODEL_IDS
from claude_router.openrouter import write_credential
from claude_router.paths import anthropic_credential_path
from claude_router.proxy import (
    LOCAL_TOKEN_HEADER,
    HybridRouterServer,
    _filter_gemini_sse_event,
    _next_sse_event,
    _remove_gemini_thinking_content,
    classify_model,
    route_payload,
)
from claude_router.storage import atomic_write_text
from claude_router.zai import write_zai_credential

OPENROUTER_KEY = "sk-or-v1-this-is-a-fake-test-key"
ANTHROPIC_KEY = "sk-ant-this-is-a-fake-test-key"
ZAI_KEY = "zai-coding-plan-test-key-0123456789"
LOCAL_TOKEN = "local-router-test-token"
GLM = "z-ai/glm-5.3-flash"
ZAI = "glm-5.3-flash"
DEEPSEEK = "~deepseek/deepseek-v4-flash-latest"
GEMINI = "google/gemini-3.8-flash"
STREAM_FIRST = b"data: first\n\n"
STREAM_SECOND = b"data: second\n\n"


class RecordingUpstream(BaseHTTPRequestHandler):
    requests: list[dict[str, object]] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length)
        self.requests.append(
            {
                "path": self.path,
                "headers": {key.casefold(): value for key, value in self.headers.items()},
                "body": json.loads(body),
            }
        )
        response = json.dumps({"type": "message", "model": json.loads(body)["model"]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, *_args) -> None:
        return


class StreamingUpstream(BaseHTTPRequestHandler):
    first_written = threading.Event()
    release_second = threading.Event()

    def do_POST(self) -> None:  # noqa: N802
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(STREAM_FIRST) + len(STREAM_SECOND)))
        self.end_headers()
        self.wfile.write(STREAM_FIRST)
        self.wfile.flush()
        self.first_written.set()
        self.release_second.wait(timeout=2)
        self.wfile.write(STREAM_SECOND)
        self.wfile.flush()

    def log_message(self, *_args) -> None:
        return


@pytest.fixture
def routing_servers(isolated_home):
    write_credential(OPENROUTER_KEY)
    atomic_write_text(anthropic_credential_path(), f"{ANTHROPIC_KEY}\n", 0o600)
    write_zai_credential(ZAI_KEY)
    RecordingUpstream.requests = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), RecordingUpstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    base = f"http://127.0.0.1:{upstream.server_port}"
    router = HybridRouterServer(
        ("127.0.0.1", 0),
        local_token=LOCAL_TOKEN,
        favorites={GLM, GEMINI, ZAI},
        anthropic_auth="max",
        anthropic_upstream=f"{base}/anthropic",
        openrouter_upstream=f"{base}/openrouter",
        zai_upstream=f"{base}/zai",
    )
    router_thread = threading.Thread(target=router.serve_forever, daemon=True)
    router_thread.start()
    try:
        yield router
    finally:
        router.shutdown()
        router.server_close()
        router_thread.join(timeout=2)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=2)


def request(
    router,
    model: str,
    *,
    authorization: str = "Bearer max-oauth",
    extra_headers: dict[str, str] | None = None,
):
    body = json.dumps({"model": model, "max_tokens": 1, "messages": []}).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": authorization,
        "X-Api-Key": "must-not-leak",
        LOCAL_TOKEN_HEADER: LOCAL_TOKEN,
    }
    headers.update(extra_headers or {})
    return urllib.request.urlopen(
        urllib.request.Request(
            f"http://127.0.0.1:{router.server_port}/v1/messages",
            data=body,
            headers=headers,
            method="POST",
        ),
        timeout=3,
    )


def test_openrouter_route_strips_cross_provider_credentials(routing_servers) -> None:
    with request(routing_servers, f"clr/openrouter/{GLM}") as response:
        assert response.status == 200
    captured = RecordingUpstream.requests[-1]
    assert captured["path"] == "/openrouter/v1/messages"
    assert captured["body"]["model"] == GLM  # type: ignore[index]
    headers = captured["headers"]
    assert headers["authorization"] == f"Bearer {OPENROUTER_KEY}"  # type: ignore[index]
    assert "x-api-key" not in headers
    assert LOCAL_TOKEN_HEADER.casefold() not in headers
    assert "max-oauth" not in json.dumps(captured)


def test_gemini_route_strips_claude_beta_header(routing_servers) -> None:
    beta = "claude-code-20250219,interleaved-thinking-2025-05-14,effort-2025-11-24"
    with request(
        routing_servers,
        f"clr/openrouter/{GEMINI}",
        extra_headers={"Anthropic-Beta": beta},
    ) as response:
        assert response.status == 200
    assert "anthropic-beta" not in RecordingUpstream.requests[-1]["headers"]


def test_zai_route_uses_zai_upstream_and_only_the_zai_key(routing_servers) -> None:
    with request(routing_servers, f"clr/zai/{ZAI}") as response:
        assert response.status == 200
    captured = RecordingUpstream.requests[-1]
    assert captured["path"] == "/zai/v1/messages"
    assert captured["body"]["model"] == ZAI  # type: ignore[index]
    headers = captured["headers"]
    assert headers["authorization"] == f"Bearer {ZAI_KEY}"  # type: ignore[index]
    assert "x-api-key" not in headers
    assert "http-referer" not in headers
    assert "x-title" not in headers
    assert OPENROUTER_KEY not in json.dumps(captured)
    assert ANTHROPIC_KEY not in json.dumps(captured)
    assert "max-oauth" not in json.dumps(captured)


def test_non_gemini_route_preserves_claude_beta_header(routing_servers) -> None:
    beta = "claude-code-20250219,interleaved-thinking-2025-05-14"
    with request(
        routing_servers,
        f"clr/openrouter/{GLM}",
        extra_headers={"Anthropic-Beta": beta},
    ) as response:
        assert response.status == 200
    assert RecordingUpstream.requests[-1]["headers"]["anthropic-beta"] == beta


def test_openrouter_stream_is_forwarded_before_upstream_finishes(isolated_home) -> None:
    write_credential(OPENROUTER_KEY)
    StreamingUpstream.first_written = threading.Event()
    StreamingUpstream.release_second = threading.Event()
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), StreamingUpstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    router = HybridRouterServer(
        ("127.0.0.1", 0),
        local_token=LOCAL_TOKEN,
        favorites={GLM},
        anthropic_auth="max",
        openrouter_upstream=f"http://127.0.0.1:{upstream.server_port}",
    )
    router_thread = threading.Thread(target=router.serve_forever, daemon=True)
    router_thread.start()
    received = threading.Event()
    result: list[bytes] = []

    def consume() -> None:
        with request(router, f"clr/openrouter/{GLM}") as response:
            result.append(response.read(len(STREAM_FIRST)))
            received.set()
            result.append(response.read())

    client = threading.Thread(target=consume, daemon=True)
    client.start()
    try:
        assert StreamingUpstream.first_written.wait(timeout=1)
        assert received.wait(timeout=0.5), "router buffered the first streaming event"
    finally:
        StreamingUpstream.release_second.set()
        client.join(timeout=2)
        router.shutdown()
        router.server_close()
        router_thread.join(timeout=2)
        upstream.shutdown()
        upstream.server_close()
        upstream_thread.join(timeout=2)
    assert result == [STREAM_FIRST, STREAM_SECOND]


def test_native_route_preserves_oauth_and_never_uses_openrouter_key(routing_servers) -> None:
    with request(routing_servers, "claude-sonnet-4-6") as response:
        assert response.status == 200
    captured = RecordingUpstream.requests[-1]
    assert captured["path"] == "/anthropic/v1/messages"
    assert captured["body"]["model"] == "claude-sonnet-4-6"  # type: ignore[index]
    headers = captured["headers"]
    assert headers["authorization"] == "Bearer max-oauth"  # type: ignore[index]
    assert "x-api-key" not in headers
    assert OPENROUTER_KEY not in json.dumps(captured)


def test_unknown_and_unfavorited_models_fail_closed(routing_servers) -> None:
    for model in ("google/gemini", "clr/openrouter/not/selected"):
        with pytest.raises(urllib.error.HTTPError) as rejected:
            request(routing_servers, model)
        assert rejected.value.code == 400
        payload = json.loads(rejected.value.read())
        assert payload["type"] == "error"
        assert payload["error"]["type"] == "invalid_request_error"
    assert RecordingUpstream.requests == []


def test_local_fallback_token_cannot_be_used_as_max_oauth(routing_servers) -> None:
    with pytest.raises(urllib.error.HTTPError) as rejected:
        request(routing_servers, "claude-opus-5", authorization=f"Bearer {LOCAL_TOKEN}")
    assert rejected.value.code == 502
    assert RecordingUpstream.requests == []


def test_anthropic_api_mode_injects_only_anthropic_key(routing_servers) -> None:
    routing_servers.anthropic_auth = "api"
    with request(routing_servers, "claude-opus-5") as response:
        assert response.status == 200
    captured = RecordingUpstream.requests[-1]
    headers = captured["headers"]
    assert headers["x-api-key"] == ANTHROPIC_KEY  # type: ignore[index]
    assert "authorization" not in headers
    assert OPENROUTER_KEY not in json.dumps(captured)


def test_healthcheck_requires_local_token(routing_servers) -> None:
    url = f"http://127.0.0.1:{routing_servers.server_port}/healthz"
    with pytest.raises(urllib.error.HTTPError) as rejected:
        urllib.request.urlopen(url, timeout=3)
    assert rejected.value.code == 401
    payload = json.loads(rejected.value.read())
    assert payload["error"]["type"] == "authentication_error"
    with urllib.request.urlopen(
        urllib.request.Request(url, headers={LOCAL_TOKEN_HEADER: LOCAL_TOKEN}), timeout=3
    ) as response:
        assert response.status == 200


def test_router_bind_does_not_wait_for_reverse_dns(isolated_home, monkeypatch) -> None:
    def fail_getfqdn(_host: str) -> str:
        raise AssertionError("router bind must not perform reverse DNS")

    monkeypatch.setattr(socket, "getfqdn", fail_getfqdn)
    router = HybridRouterServer(
        ("127.0.0.1", 0),
        local_token=LOCAL_TOKEN,
        favorites={GLM},
        anthropic_auth="max",
    )
    try:
        assert router.server_name == "127.0.0.1"
        assert router.server_port > 0
    finally:
        router.server_close()


def test_model_classification_is_explicit() -> None:
    assert classify_model("claude-opus-5", {GLM}) == ("anthropic", "claude-opus-5")
    assert classify_model(f"clr/openrouter/{GLM}", {GLM}) == ("openrouter", GLM)
    assert classify_model(f"clr/zai/{ZAI}", {ZAI}) == ("zai", ZAI)
    with pytest.raises(ValueError, match="no trusted route"):
        classify_model(GLM, {GLM})
    with pytest.raises(ValueError, match="blocked on the OpenRouter route"):
        classify_model("clr/openrouter/anthropic/claude-opus-5", {"anthropic/claude-opus-5"})


def test_zai_classification_requires_the_favorites_allowlist() -> None:
    assert ZAI in ZAI_MODEL_IDS
    assert classify_model(f"clr/zai/{ZAI}", {ZAI}) == ("zai", ZAI)
    with pytest.raises(ValueError, match="Z.ai model is not in the clr favorites allowlist"):
        classify_model(f"clr/zai/{ZAI}", set())


def test_zai_route_payload_rewrites_model_and_keeps_other_fields() -> None:
    payload = {
        "model": f"clr/zai/{ZAI}",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "hello"}],
    }

    route, model, body = route_payload(json.dumps(payload).encode(), {ZAI})
    routed = json.loads(body)

    assert (route, model, routed["model"]) == ("zai", ZAI, ZAI)
    # default effort=high raises the token floor so thinking cannot truncate
    assert routed["max_tokens"] == 8192
    assert routed["messages"] == payload["messages"]


def test_zai_route_strips_unicode_property_class_patterns() -> None:
    payload = {
        "model": f"clr/zai/{ZAI}",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [
            {
                "name": "artifact",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "field": {
                            "type": "string",
                            "pattern": "^[^\\p{Cc}]+$",
                            "description": "untouched",
                        },
                        "plain": {"type": "string", "pattern": "^[a-z]+$"},
                    },
                },
            }
        ],
    }

    route, model, body = route_payload(json.dumps(payload).encode(), {ZAI})
    routed = json.loads(body)
    props = routed["tools"][0]["input_schema"]["properties"]

    assert (route, model) == ("zai", ZAI)
    assert "pattern" not in props["field"]
    assert props["field"]["description"] == "untouched"
    assert props["plain"]["pattern"] == "^[a-z]+$"


def test_non_zai_route_keeps_unicode_property_class_patterns() -> None:
    payload = {
        "model": f"clr/openrouter/{GLM}",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [
            {
                "name": "artifact",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string", "pattern": "^[^\\p{Cc}]+$"},
                    },
                },
            }
        ],
    }

    _, _, body = route_payload(json.dumps(payload).encode(), {GLM})
    routed = json.loads(body)

    pattern = routed["tools"][0]["input_schema"]["properties"]["field"]["pattern"]
    assert pattern == "^[^\\p{Cc}]+$"


def test_zai_text_only_modality_handling_matches_openrouter() -> None:
    payload = {
        "model": f"clr/zai/{ZAI}",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": "abc"},
                    }
                ],
            }
        ],
    }

    route, model, body = route_payload(
        json.dumps(payload).encode(), {ZAI}, {ZAI: frozenset({"text"})}
    )
    routed = json.loads(body)

    assert (route, model) == ("zai", ZAI)
    replacement = routed["messages"][0]["content"][0]
    assert replacement["type"] == "text"
    assert "InputError[unsupported_input_modality]" in replacement["text"]
    assert '"data":"abc"' not in body.decode()


def test_text_only_model_receives_capability_notice_and_failed_image_tool_result() -> None:
    image_data = "must-not-reach-openrouter"
    payload = {
        "model": f"clr/openrouter/{DEEPSEEK}",
        "system": [{"type": "text", "text": "You are a coding agent."}],
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_read_image",
                        "name": "Read",
                        "input": {"file_path": "/tmp/image.jpg"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_read_image",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_data,
                                },
                            }
                        ],
                    }
                ],
            },
        ],
    }
    route, model, body = route_payload(
        json.dumps(payload).encode(),
        {DEEPSEEK, GLM},
        {DEEPSEEK: frozenset({"text"}), GLM: frozenset({"text", "image", "video"})},
    )
    routed = json.loads(body)

    assert (route, model, routed["model"]) == ("openrouter", DEEPSEEK, DEEPSEEK)
    assert image_data not in body.decode()
    notice = routed["system"][-1]["text"]
    assert "text-only" in notice
    assert "cannot inspect image pixels" in notice
    tool_result = routed["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert tool_result["tool_use_id"] == "toolu_read_image"
    error = tool_result["content"][0]["text"]
    assert "ToolError[unsupported_input_modality]" in error
    assert "Do not retry" in error
    assert GLM in error


def test_text_only_model_replaces_direct_image_with_categorized_input_error() -> None:
    payload = {
        "model": f"clr/openrouter/{DEEPSEEK}",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": "abc"},
                    },
                ],
            }
        ],
    }
    _, _, body = route_payload(
        json.dumps(payload).encode(),
        {DEEPSEEK},
        {DEEPSEEK: frozenset({"text"})},
    )
    routed = json.loads(body)
    replacement = routed["messages"][0]["content"][1]

    assert replacement["type"] == "text"
    assert "InputError[unsupported_input_modality]" in replacement["text"]
    assert '"data":"abc"' not in body.decode()


def test_vision_model_keeps_image_payload_unchanged() -> None:
    payload = {
        "model": f"clr/openrouter/{GLM}",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": "abc"},
                    }
                ],
            }
        ],
    }
    _, _, body = route_payload(
        json.dumps(payload).encode(),
        {GLM},
        {GLM: frozenset({"text", "image", "video"})},
    )
    routed = json.loads(body)

    assert routed["messages"][0]["content"][0]["type"] == "image"
    assert routed["messages"][0]["content"][0]["source"]["data"] == "abc"
    assert "system" not in routed


def test_gemini_route_repairs_nested_itemless_tool_arrays() -> None:
    payload = {
        "model": f"clr/openrouter/{GEMINI}",
        "thinking": {"type": "adaptive", "display": "omitted"},
        "output_config": {"effort": "high"},
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [
            {
                "name": "query_records",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "object",
                            "properties": {
                                "where": {
                                    "type": "array",
                                    "items": {"type": "array"},
                                }
                            },
                        }
                    },
                },
            }
        ],
    }

    route, model, body = route_payload(json.dumps(payload).encode(), {GEMINI})
    routed = json.loads(body)
    where = routed["tools"][0]["input_schema"]["properties"]["query"]["properties"]["where"]

    assert (route, model, routed["model"]) == ("openrouter", GEMINI, GEMINI)
    assert "thinking" not in routed
    assert "output_config" not in routed
    assert where["items"] == {"type": "array", "items": {"type": "string"}}


def test_non_gemini_route_keeps_open_ended_tool_schema() -> None:
    payload = {
        "model": f"clr/openrouter/{GLM}",
        "thinking": {"type": "adaptive", "display": "omitted"},
        "output_config": {"effort": "high"},
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [
            {
                "name": "open_array",
                "input_schema": {
                    "type": "object",
                    "properties": {"values": {"type": "array"}},
                },
            }
        ],
    }

    _, _, body = route_payload(json.dumps(payload).encode(), {GLM})

    routed = json.loads(body)
    assert "items" not in routed["tools"][0]["input_schema"]["properties"]["values"]
    assert routed["thinking"] == {"type": "adaptive", "display": "omitted"}
    assert routed["output_config"] == {"effort": "high"}


def test_gemini_sse_filter_removes_thinking_block_and_signature() -> None:
    events = [
        b'data: {"type":"content_block_start","index":0,'
        b'"content_block":{"type":"text","text":""}}\n\n',
        b'data: {"type":"content_block_delta","index":0,'
        b'"delta":{"type":"text_delta","text":"hello"}}\n\n',
        b'data: {"type":"content_block_start","index":1,'
        b'"content_block":{"type":"thinking","thinking":"","signature":""}}\n\n',
        b'data: {"type":"content_block_stop","index":0}\n\n',
        b'data: {"type":"content_block_delta","index":1,'
        b'"delta":{"type":"signature_delta","signature":"secret"}}\n\n',
        b'data: {"type":"content_block_stop","index":1}\n\n',
    ]
    thinking_indexes: set[int] = set()

    filtered = b"".join(_filter_gemini_sse_event(event, thinking_indexes) for event in events)

    assert b'"text":"hello"' in filtered
    assert b'"index":0' in filtered
    assert b'"index":1' not in filtered
    assert b"thinking" not in filtered
    assert b"signature" not in filtered
    assert thinking_indexes == set()


def test_sse_splitter_handles_lf_and_crlf_boundaries() -> None:
    first = b"data: first\n\n"
    second = b"data: second\r\n\r\n"
    event, remainder = _next_sse_event(first + second) or (b"", b"")
    assert event == first
    assert _next_sse_event(remainder) == (second, b"")


def test_gemini_non_streaming_response_drops_thinking_content() -> None:
    body = json.dumps(
        {
            "type": "message",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "thinking", "thinking": "hidden", "signature": "secret"},
            ],
        }
    ).encode()

    normalized = json.loads(_remove_gemini_thinking_content(body))

    assert normalized["content"] == [{"type": "text", "text": "hello"}]


def test_zai_route_strips_context_budget_suffix() -> None:
    payload = {
        "model": f"clr/zai/{ZAI}[1m]",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hello"}],
    }

    route, model, body = route_payload(json.dumps(payload).encode(), {ZAI})
    routed = json.loads(body)

    assert (route, model, routed["model"]) == ("zai", ZAI, ZAI)


def test_openrouter_route_strips_context_budget_suffix() -> None:
    payload = {
        "model": f"clr/openrouter/{GLM}[1m]",
        "messages": [{"role": "user", "content": "hello"}],
    }

    route, model, body = route_payload(json.dumps(payload).encode(), {GLM})
    routed = json.loads(body)

    assert (route, model, routed["model"]) == ("openrouter", GLM, GLM)


def test_wafer_classification_and_route() -> None:
    assert classify_model("clr/wafer/GLM-5.3", {"GLM-5.3"}) == ("wafer", "GLM-5.3")
    with pytest.raises(ValueError, match="Wafer model is not in the clr favorites allowlist"):
        classify_model("clr/wafer/GLM-5.3", set())


def test_wafer_route_payload_rewrites_model_and_strips_patterns() -> None:
    payload = {
        "model": "clr/wafer/GLM-5.3[1m]",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [
            {
                "name": "artifact",
                "input_schema": {
                    "type": "object",
                    "properties": {"field": {"type": "string", "pattern": "^[^\\p{Cc}]+$"}},
                },
            }
        ],
    }

    route, model, body = route_payload(json.dumps(payload).encode(), {"GLM-5.3"})
    routed = json.loads(body)
    props = routed["tools"][0]["input_schema"]["properties"]

    assert (route, model, routed["model"]) == ("wafer", "GLM-5.3", "GLM-5.3")
    assert "pattern" not in props["field"]


def test_fireworks_classification_and_route() -> None:
    assert classify_model("clr/fireworks/glm-5p2", {"glm-5p2"}) == ("fireworks", "glm-5p2")
    with pytest.raises(ValueError, match="Fireworks model is not in the clr favorites allowlist"):
        classify_model("clr/fireworks/glm-5p2", set())


def test_fireworks_route_payload_rewrites_model() -> None:
    payload = {
        "model": "clr/fireworks/deepseek-v4-pro-0813",
        "messages": [{"role": "user", "content": "hello"}],
    }

    route, model, body = route_payload(json.dumps(payload).encode(), {"deepseek-v4-pro-0813"})
    routed = json.loads(body)

    expected = "deepseek-v4-pro-0813"
    assert (route, model, routed["model"]) == ("fireworks", expected, expected)


def test_fireworks_service_tier_injected_when_configured() -> None:
    payload = {
        "model": "clr/fireworks/glm-5p3-flash",
        "messages": [{"role": "user", "content": "hello"}],
    }

    route, model, body = route_payload(
        json.dumps(payload).encode(),
        {"glm-5p3-flash"},
        fireworks_service_tier="priority",
    )
    routed = json.loads(body)

    assert route == "fireworks"
    assert routed["service_tier"] == "priority"


def test_fireworks_service_tier_omitted_by_default() -> None:
    payload = {
        "model": "clr/fireworks/glm-5p3-flash",
        "messages": [{"role": "user", "content": "hello"}],
    }

    _, _, body = route_payload(json.dumps(payload).encode(), {"glm-5p3-flash"})
    routed = json.loads(body)

    assert "service_tier" not in routed


def test_inco_classification_and_route() -> None:
    assert classify_model("clr/inco/GLM-5.3", {"GLM-5.3"}) == ("inco", "GLM-5.3")
    with pytest.raises(ValueError, match="Inco model is not in the clr favorites allowlist"):
        classify_model("clr/inco/GLM-5.3", set())


def test_effort_high_stamped_by_default_on_flash_routes() -> None:
    payload = {
        "model": f"clr/zai/{ZAI}",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "hello"}],
    }

    _, _, body = route_payload(json.dumps(payload).encode(), {ZAI})
    routed = json.loads(body)

    assert routed["thinking"] == {"type": "adaptive"}
    assert routed["output_config"]["effort"] == "high"
    assert routed["max_tokens"] == 8192  # raised to the high floor


def test_effort_respects_client_thinking_and_overrides() -> None:
    with_thinking = {
        "model": f"clr/zai/{ZAI}",
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": "hello"}],
    }
    _, _, body = route_payload(json.dumps(with_thinking).encode(), {ZAI})
    routed = json.loads(body)
    assert routed["thinking"] == {"type": "disabled"}  # untouched
    assert "output_config" not in routed

    override = {
        "model": "clr/fireworks/glm-5p3-flash",
        "max_tokens": 200,
        "messages": [{"role": "user", "content": "hello"}],
    }
    _, _, body = route_payload(
        json.dumps(override).encode(), {"glm-5p3-flash"},
        effort_overrides={"fireworks": "low"},
    )
    routed = json.loads(body)
    assert routed["output_config"]["effort"] == "low"
    assert routed["max_tokens"] == 200  # low keeps the client cap

    _, _, body = route_payload(
        json.dumps(override).encode(), {"glm-5p3-flash"},
        effort_overrides={"fireworks": "off"},
    )
    routed = json.loads(body)
    assert "output_config" not in routed and "thinking" not in routed


def test_effort_not_stamped_on_non_flash_routes() -> None:
    payload = {
        "model": "claude-fable-5-1",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "hello"}],
    }
    _, _, body = route_payload(json.dumps(payload).encode(), set())
    routed = json.loads(body)
    assert "thinking" not in routed and "output_config" not in routed


def test_effort_inco_uses_hidden_switch_only_for_low() -> None:
    low = {"model": "clr/inco/glm-5.3-flash:fast", "messages": [{"role": "user", "content": "x"}]}
    _, _, body = route_payload(json.dumps(low).encode(), {"glm-5.3-flash:fast"})
    routed = json.loads(body)
    assert routed["reasoning"] == {"effort": "low"} if "effort" in str(
        json.dumps(routed)
    ) else True  # default high injects nothing on inco
    assert "output_config" not in routed
    assert routed.get("reasoning") is None  # default high -> natural state

    forced = route_payload(
        json.dumps(low).encode(), {"glm-5.3-flash:fast"},
        effort_overrides={"inco": "low"},
    )[2]
    routed = json.loads(forced)
    assert routed["reasoning"] == {"effort": "low"}


def test_effort_wafer_low_maps_to_thinking_disabled() -> None:
    payload = {"model": "clr/wafer/GLM-5.3-Flash", "messages": [{"role": "user", "content": "x"}]}
    _, _, body = route_payload(
        json.dumps(payload).encode(), {"GLM-5.3-Flash"},
        effort_overrides={"wafer": "low"},
    )
    routed = json.loads(body)
    assert routed["thinking"] == {"type": "disabled"}
    assert "output_config" not in routed

    # default high injects nothing on wafer
    _, _, body = route_payload(json.dumps(payload).encode(), {"GLM-5.3-Flash"})
    routed = json.loads(body)
    assert "thinking" not in routed and "output_config" not in routed
