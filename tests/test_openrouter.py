from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from claude_openrouter.models import ZAI_MODELS
from claude_openrouter.openrouter import (
    load_catalog,
    refresh_catalog,
    save_catalog,
    validate_key,
    validate_key_shape,
    write_credential,
)
from claude_openrouter.paths import catalog_path, credential_path
from claude_openrouter.storage import atomic_write_text, read_json_object

KEY = "sk-or-v1-this-is-a-fake-test-key"


def test_key_shape_and_private_write(isolated_home) -> None:
    validate_key_shape(KEY)
    with pytest.raises(ValueError):
        validate_key_shape("anthropic-key")
    write_credential(KEY)
    assert credential_path().read_text().strip() == KEY
    assert credential_path().stat().st_mode & 0o777 == 0o600


def test_validate_and_refresh_use_bearer_and_persist(
    isolated_home, sample_models, monkeypatch
) -> None:
    requests: list[tuple[str, str | None]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            requests.append((self.path, self.headers.get("Authorization")))
            payload = (
                {"data": {"label": "test"}} if self.path == "/key" else {"data": sample_models}
            )
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("CLAUDE_OPENROUTER_API_BASE", f"http://127.0.0.1:{server.server_port}")
    try:
        validate_key(KEY)
        assert refresh_catalog(KEY) == [*sample_models, *ZAI_MODELS]
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert requests == [
        ("/key", f"Bearer {KEY}"),
        ("/models", f"Bearer {KEY}"),
    ]
    assert load_catalog() == [*sample_models, *ZAI_MODELS]
    assert catalog_path().stat().st_mode & 0o777 == 0o600


def test_load_catalog_appends_zai_models_without_caching_them(isolated_home, sample_models) -> None:
    save_catalog(sample_models)

    assert read_json_object(catalog_path())["models"] == sample_models
    catalog = load_catalog()
    assert catalog[: len(sample_models)] == sample_models
    assert catalog[len(sample_models) :] == ZAI_MODELS
    assert "glm-5.3-flash" in {str(model["id"]) for model in catalog}


def test_load_catalog_without_an_index_returns_exactly_the_zai_models(isolated_home) -> None:
    assert not catalog_path().exists()
    assert load_catalog() == ZAI_MODELS


def test_refresh_catalog_without_a_credential_needs_no_network(isolated_home, monkeypatch) -> None:
    monkeypatch.setattr(
        "claude_openrouter.openrouter.fetch_models",
        lambda _key: (_ for _ in ()).throw(AssertionError("fetch_models must not be called")),
    )

    assert not credential_path().exists()
    assert refresh_catalog() == ZAI_MODELS


def test_refresh_catalog_with_a_malformed_credential_still_raises(isolated_home) -> None:
    atomic_write_text(credential_path(), "not-an-openrouter-key\n")

    with pytest.raises(ValueError, match="unexpected format"):
        refresh_catalog()
