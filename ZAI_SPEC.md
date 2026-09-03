# Task: add a Z.ai (GLM Coding Plan) route to claude-openrouter

You are working in a fork of `claude-openrouter`, a small dependency-free Python
package (`src/claude_openrouter/`) that runs a loopback router on
`127.0.0.1:9417`. Claude Code sends every request to the router. Native Claude
model ids are forwarded to `https://api.anthropic.com` with the user's own OAuth
bearer. Favorite OpenRouter models, addressed by the private namespace
`clor/openrouter/<id>`, are forwarded to `https://openrouter.ai/api` with the
OpenRouter key. Favorites also appear as rows in Claude Code's `/model` picker
and as generated subagents in `~/.claude/agents/clor-*.md`.

Goal: add a **third route, `zai`**, so the user can pick GLM models billed to
their Z.ai Coding Plan subscription, in the same session as native Claude and
OpenRouter models. Z.ai exposes an Anthropic-compatible endpoint, so **no
request or response translation is needed**, just routing and auth.

Read the existing code before changing anything. Follow its style exactly:
type hints, `from __future__ import annotations`, no new dependencies, fail
closed on anything unknown, secrets in 0600 files, ruff-clean (line length 100).

## Facts you need

- Z.ai Anthropic-compatible upstream: `https://api.z.ai/api/anthropic`.
  Requests go to `<upstream>/v1/messages` (the router already appends the
  request path to the upstream base in `_forward`).
- Auth header for Z.ai: `Authorization: Bearer <zai key>`. Strip any
  `x-api-key` and the incoming Claude OAuth bearer, exactly as the openrouter
  route does. Never send the Z.ai key to Anthropic or OpenRouter, and never
  send the OpenRouter key or the OAuth bearer to Z.ai.
- Z.ai model ids have **no slash**: `glm-5.3`, `glm-5.3-flash`,
  `glm-5.3-highspeed`, `glm-5.2`, `glm-5-turbo`, `glm-4.7`. OpenRouter ids
  always contain a slash (`vendor/model`). Use this to keep the two catalogs
  from colliding.
- Z.ai key shape: be lenient. Accept any single token with no whitespace and at
  least 20 characters. Do not hardcode a stricter pattern.
- Z.ai models support tools and tool_choice. Context: 200000 for glm-4.7 and
  glm-5-turbo, 1000000 for the others. No per-token pricing (subscription), so
  the picker description must not show a price.

## Design (implement exactly this)

### 1. `models.py`
- Add `ZAI_MODEL_PREFIX = "clor/zai/"` beside `OPENROUTER_MODEL_PREFIX`.
- Add a static `ZAI_MODELS: list[dict[str, Any]]` with one catalog-shaped dict
  per model id above. Each dict must have: `"id"`, `"name"` (e.g. `"GLM-5.3
  Flash"`), `"description"`, `"provider": "zai"`, `"context_length"`,
  `"supported_parameters": ["tools", "tool_choice"]`, and
  `"architecture": {"input_modalities": ["text"]}` (text only; use
  `["text", "image"]` only if you are certain a model has vision. When unsure,
  text only).
- Add `ZAI_MODEL_IDS = frozenset(m["id"] for m in ZAI_MODELS)`.
- Add `def provider_of(model_id: str) -> str` returning `"zai"` if
  `model_id in ZAI_MODEL_IDS` else `"openrouter"`.
- Change `namespaced_model(model_id)` to use the zai prefix when
  `provider_of(model_id) == "zai"`.
- Change `original_model(model_id)` to strip **either** prefix and return the
  bare id, or `None`. Add `def route_of_namespaced(model_id: str) -> str | None`
  returning `"zai"`, `"openrouter"`, or `None` based on which prefix is present.
- `picker_description`: for provider zai, the second part must be
  `"Z.ai Coding Plan via claude-openrouter"` and there is no pricing part.
  For openrouter it stays `"OpenRouter via claude-openrouter"`.
- `picker_row(..., hybrid=True)`: label suffix `" · Z.ai"` for zai models,
  `" · OpenRouter"` otherwise.
- `hybrid_openrouter_allowed` stays as is (zai ids pass it).
- `exact_models` error text: change "current OpenRouter index" to
  "current model index".

### 2. `openrouter.py`
- `load_catalog()` and `refresh_catalog()` must return the OpenRouter catalog
  **plus** `ZAI_MODELS` appended (so `search`, `select`, `setup --models`,
  `exact_models`, the picker, and `configure_claude` all see GLM models with
  zero further changes). Do not write ZAI_MODELS into the cached
  `models.json`; merge at load time. Guard against duplicate ids.

### 3. New module `zai.py`
Mirror `anthropic.py`:
- `ZAI_UPSTREAM = "https://api.z.ai/api/anthropic"`
- `validate_zai_key_shape(key)`, `read_zai_credential()`,
  `write_zai_credential(key)` using a new `zai_credential_path()` in
  `paths.py` (`config_dir() / "zai-credential"`).
- `read_zai_credential` raises `RuntimeError("Z.ai credential not found at
  ...; run `clor config --zai-key`")` when missing.

### 4. `proxy.py`
- Import `ZAI_UPSTREAM` and `read_zai_credential`.
- `classify_model(model, favorites)`: if `route_of_namespaced(model) == "zai"`,
  require the bare id to be in `favorites` (else `ValueError("Z.ai model is
  not in the clor favorites allowlist")`) and return `("zai", bare_id)`.
  Keep the existing openrouter and anthropic branches unchanged.
- `route_payload`: for route `"zai"`, set `payload["model"] = upstream_model`
  and re-serialize the body. Do not apply Gemini repairs. Apply the same
  image-modality handling as openrouter if the model's modalities are known
  and exclude image.
- `HybridRouterServer.__init__`: add `zai_upstream: str = ZAI_UPSTREAM` and
  store it. In the request handler choose the upstream by route:
  openrouter / zai / anthropic.
- `_upstream_headers`: for route `"zai"`, set
  `headers["Authorization"] = f"Bearer {read_zai_credential()}"`. Do not add
  `HTTP-Referer` / `X-Title`. Keep `anthropic-version` and `anthropic-beta`
  headers from the incoming request.
- `_record_status` already takes a route string; make sure `"zai"` flows
  through unchanged.

### 5. `settings.py`
- `_looks_managed_picker`: match `"via claude-openrouter"` in the description
  instead of `"OpenRouter via claude-openrouter"`, so zai rows count as
  managed and `clor reset` removes them.
- `assert_private_files`: include `zai_credential_path()`.
- `configure_claude`: if any selected model has `provider == "zai"` and
  `zai_credential_path()` does not exist, raise
  `RuntimeError("Z.ai favorites require a configured Z.ai key; run `clor
  config --zai-key`")`.

### 6. `agents.py`
- `_agent_document`: description must say `"Z.ai model"` instead of
  `"OpenRouter favorite"` for zai models, and the body sentence should say
  "configured Z.ai model" for them. Keep the managed marker and everything
  else identical.

### 7. `cli.py`
- `config`: add `--zai-key` (flag, interactive masked prompt like the others,
  label `"Z.ai API key"`) and `--zai-key-stdin`. Either writes the credential
  via `write_zai_credential`, runs `assert_private_files()`, prints
  `Z.ai credential updated: <path> (mode 0600)`, and returns 0. They cannot
  be combined with the OpenRouter or Anthropic credential options in one
  invocation (raise `ValueError` like the existing combination checks).
- `setup`: add the same two options as optional. If given, write the Z.ai
  credential before `configure_claude`. Setup must still work with no Z.ai
  key when no zai model is selected.
- `command_check`: for a zai model id, skip the OpenRouter cost estimate
  (print `Estimated charge: billed to your Z.ai Coding Plan quota.`) and
  otherwise run the same `probe_model` flow. Look at `check.py` to see how
  the probe addresses the model and make sure it uses the namespaced id so
  the router picks the zai route.
- `command_doctor`: add `"zai_credential": bool` to the status dict, print a
  `Z.ai credential:` line (`configured` / `missing`), and mark the doctor
  unhealthy only if a zai favorite is selected and the credential is missing.
- `command_setup` summary: the "Next:" line should say "native, OpenRouter,
  and Z.ai models".
- Update the argparse `description` and the `check` help text so they no
  longer say OpenRouter-only.

### 8. Tests (`tests/`)
Add tests, following the style of the existing files and reusing
`conftest.py` fixtures:
- `test_models.py`: `provider_of`, `namespaced_model`/`original_model`
  round-trip for both prefixes, `route_of_namespaced`, zai `picker_row`
  label and description (no price, contains "Z.ai Coding Plan").
- `test_proxy.py`: `classify_model` returns `("zai", "glm-5.3-flash")` for a
  favorited zai model, raises for a non-favorited one, and still handles
  openrouter and anthropic; `route_payload` rewrites the model field for zai;
  the handler picks `zai_upstream` and sends `Authorization: Bearer <zai
  key>` without the OAuth bearer or OpenRouter key (look at how existing
  proxy tests fake upstreams and credentials, and do the same).
- `test_settings.py`: `configure_claude` with a zai favorite writes a picker
  row with the `clor/zai/` model id and raises without a Z.ai credential;
  `_looks_managed_picker` recognises a zai row.
- `test_cli.py`: `config --zai-key-stdin` writes the 0600 credential file.
- `test_openrouter.py`: `load_catalog()` includes the zai models.

### 9. README
Add a short section "Z.ai Coding Plan models" explaining: `clor config
--zai-key`, then `clor select glm-5.3-flash` (or pick them in the picker),
how they show up as `· Z.ai` rows, and that they bill to the Z.ai
subscription. Update the architecture diagram to show the third arrow.

## Verification, mandatory before you finish
Run, from the repo root:

    uv run --with pytest --with ruff -q ruff check src tests
    uv run --with pytest --with ruff -q ruff format --check src tests
    uv run --with pytest -q pytest -q

All new tests must pass. One pre-existing failure,
`tests/test_service.py::test_startup_failure_includes_router_log`, is
environment-specific and is allowed to keep failing; nothing else may fail.

Then run this quick manual routing check without any network:

    uv run python -c "from claude_openrouter.proxy import classify_model; print(classify_model('clor/zai/glm-5.3-flash', {'glm-5.3-flash'})); print(classify_model('claude-opus-4-8', set()))"

Expected: `('zai', 'glm-5.3-flash')` then `('anthropic', 'claude-opus-4-8')`.

## Rules
- Do not commit. Leave the changes in the working tree.
- Do not touch `~/.claude`, `~/.config/claude-openrouter`, or start the
  router service. Tests must use temp dirs, as the existing tests do.
- Do not add dependencies. Do not rename existing public functions.
- When done, print a summary: files changed, tests added, and the output of
  the three verification commands.
