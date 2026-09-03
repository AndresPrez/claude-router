# Pass 2: make OpenRouter optional so Z.ai-only setups work

Pass 1 (ZAI_SPEC.md) is done and verified. Do not redo it. Read the current
working tree first (`git diff` shows pass 1). This pass fixes one gap: a user
who only has a Z.ai Coding Plan and no OpenRouter account cannot complete
`clor setup`, because setup always demands an OpenRouter key and fetches the
OpenRouter catalog.

## Required behaviour

1. `clor setup --no-openrouter`
   - New flag. When set, do not prompt for or validate an OpenRouter key, do
     not call `validate_key`, and do not fetch the OpenRouter catalog. The
     model list offered to the picker (or validated for `--models`) is
     `merged_catalog([])`, i.e. only the static `ZAI_MODELS`.
   - `--no-openrouter` cannot be combined with `--key-stdin` (ValueError).
   - If neither `--zai-key` nor `--zai-key-stdin` is given and no Z.ai
     credential exists yet, prompt for the Z.ai key (call `_read_zai_key`),
     because with no OpenRouter key the only useful favorites are Z.ai ones.
   - The setup summary must not print the "OpenRouter credential:" line when
     no OpenRouter credential exists; print a "Z.ai credential:" line when one
     exists (path plus "(mode 0600)"), for both modes.

2. Catalog loading without an OpenRouter key
   - `openrouter.load_catalog()`: when the cached index file is missing,
     return `merged_catalog([])` instead of raising. Keep raising for a file
     that exists but is invalid.
   - `openrouter.refresh_catalog(key=None)`: when `key` is None and
     `read_credential()` raises because the credential file is missing,
     return `merged_catalog([])` instead of raising. A present but malformed
     credential must still raise.
   - `clor select`, `clor claude`, `clor config --anthropic-auth`, and the
     router's startup path (`run_router` / wherever it loads the catalog and
     modalities) must therefore work with only Z.ai favorites and no
     OpenRouter credential. Check each call site and add a test for `select`
     with a Z.ai model and no OpenRouter credential.

3. `clor check <zai-model>` must not require an OpenRouter key. Look at how
   `command_check` obtains the model dict and make sure it works from
   `merged_catalog([])` when the model is a Z.ai id.

4. `clor doctor`
   - `openrouter_credential` is only required for health when at least one
     favorite has provider `openrouter`. Mirror the existing `zai_selected`
     logic with an `openrouter_selected` flag.
   - The human-readable output should show both credential lines with
     configured / missing / not needed.

5. `clor search` and `clor index` with no OpenRouter credential: print the
   Z.ai models only and a one-line stderr note
   `note: no OpenRouter credential; showing Z.ai models only`. Do not error.

6. Tests, in the existing style with the conftest fixtures:
   - `setup --no-openrouter --zai-key-stdin --models glm-5.3-flash` succeeds
     with no OpenRouter credential, writes the Z.ai credential, and produces a
     picker with only the `clor/zai/glm-5.3-flash` row. (Look at how existing
     setup tests stub `has_native_login`, `start_service`, and
     `configure_claude` inputs, and do the same.)
   - `setup --no-openrouter --key-stdin` raises ValueError.
   - `load_catalog()` with no index file returns exactly the Z.ai models.
   - `refresh_catalog()` with no credential file returns exactly the Z.ai
     models and does not hit the network (assert `fetch_models` is not
     called).
   - `doctor` is healthy with a Z.ai favorite, a Z.ai credential, and no
     OpenRouter credential.

7. README: in the "Z.ai Coding Plan models" section, add the Z.ai-only quick
   start: `clor setup --no-openrouter`.

## Verification, mandatory

    uv run --with pytest --with ruff -q ruff check src tests
    uv run --with pytest --with ruff -q ruff format --check src tests
    uv run --with pytest -q pytest -q

Everything must pass except the known pre-existing failure
`tests/test_service.py::test_startup_failure_includes_router_log`.

## Rules
- Run `ruff format` ONLY on files you changed in this pass, never on the
  whole tree. Do not reformat files you did not otherwise edit.
- Do not commit. Do not touch `~/.claude` or `~/.config/claude-openrouter`.
  Do not start the router service. No new dependencies.
- Finish with a summary: files changed, tests added, and the output of the
  three verification commands.
