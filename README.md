<h1 align="center">Claude Router</h1>

<p align="center"><strong>A loopback router for Claude Code: native Claude plus Z.ai, Wafer, Fireworks, Inco, OpenRouter, and Cursor Cloud Agents — with per-request metrics.</strong></p>

Claude Router is a small, dependency-free CLI and loopback proxy. Your native
Claude login keeps working for Claude models, and favorites from every
configured provider appear beside them in Claude Code's `/model` picker. After
setup, run the normal `claude` command.

## Why

Claude Code speaks one wire protocol. Most inference providers expose it too —
either natively or via an Anthropic-compatible gateway — but each has its own
key, quirks, and speed. Claude Router terminates that complexity at loopback:

- **One picker, every provider.** Z.ai Coding Plan, Wafer Serverless, Fireworks
  AI, Inco AI, OpenRouter, and Cursor's Cloud Agents API all appear as
  namespaced models (`clr/zai/...`, `clr/wafer/...`, `clr/fireworks/...`,
  `clr/inco/...`, `clr/cursor/...`) next to your native Claude models.
- **Provider keys never touch Claude Code.** Credentials live in private
  `0600` files under the config dir; the router swaps in upstream auth at
  forwarding time. Claude Code only holds a loopback router token.
- **Schemas repaired in flight.** Upstream validators reject some valid JSON
  Schema features (Z.ai 1210 on Unicode-property `pattern` regexes, Gemini on
  itemless arrays); the router repairs them per route before forwarding.
- **Context budgets that tell the truth.** Claude Code's `[1m]` suffix marks a
  1M context budget but some upstreams reject it on the wire (Z.ai 1211); the
  router strips it before forwarding so picker rows, defaults, and subagent
  definitions can carry honest budgets.
- **Deferred tool loading works.** `ENABLE_TOOL_SEARCH` is written as `true`
  (verified against Z.ai's request shape), keeping hundreds of tool schemas
  out of every request.
- **Metrics out of the box.** Every routed request is recorded — tokens, cache
  read/write, time-to-first-byte, tokens/sec — with `clr metrics`,
  `--histogram`, and `--model` filtering to read it back.

## Routes

| Route | Upstream | Notes |
| --- | --- | --- |
| native | api.anthropic.com | Your Claude Max/API login, passed through untouched |
| `clr/zai/<model>` | api.z.ai/api/anthropic | Coding Plan quota; GLM-5.3 and GLM-5.3-Flash |
| `clr/wafer/<model>` | pass.wafer.ai | Self-hosted GLM-5.3 at the top of the speed table |
| `clr/fireworks/<model>` | api.fireworks.ai/inference | GLM/Kimi/DeepSeek; `*-fast` routers; optional `fireworks_service_tier` preference |
| `clr/inco/<model>` | api.inco.ai | DFlash speculative decoding; `:fast` model variants |
| `clr/cursor/<model>` | Cloud Agents API | Durable cloud agents; run SSE translated to Anthropic shapes; usage fetched post-run |
| `clr/openrouter/<model>` | openrouter.ai | The original route, kept working |

Routes are favorites-gated: only models you explicitly enable can be billed.

## Install

```bash
uv tool install git+https://github.com/AndresPrez/claude-router
```

Or with curl:

```bash
curl -LsSf https://andresperez.github.io/claude-router/install.sh | sh
```

## Quick start

```bash
clr setup --no-openrouter --zai-key   # Z.ai Coding Plan only
```

Add other providers as you go — each stores a private credential and enables
its favorites:

```bash
clr config --wafer-key
clr config --fireworks-key
clr config --inco-key
clr config --cursor-key
clr config --zai-key
```

Then run `claude` and switch models with `/model`. Built-in Claude models keep
using your native login; rows labeled `· Z.ai`, `· Wafer`, `· Fireworks`, `· Inco`,
`· Cursor`, or `· OpenRouter` bill to the matching provider.

## Metrics

```bash
clr metrics                    # per-model totals, cache hit/write volumes, tok/s, TTFT
clr metrics --histogram        # requests per hour, segmented by route
clr metrics --model flash      # filter by model substring
clr metrics --days 30 --json   # machine-readable
```

Records live in `metrics.jsonl` under the state directory (rotated at 10 MB).
The passthrough routes tap usage from the response stream without altering it;
the Cursor bridge folds in usage from the run usage endpoint, including on
cancelled runs.

## Commands

| Command | Purpose |
| --- | --- |
| `clr index` / `clr fetch` | Fetch and cache the current OpenRouter model catalog |
| `clr search QUERY...` | Refresh, then search names, IDs, and descriptions |
| `clr search QUERY... --tools` | Show only models advertising tool calling |
| `clr check MODEL [--yes]` | Confirm and run one billable Claude Code tool round-trip |
| `clr setup` | Run the install-time key and model setup again |
| `clr select [MODEL]` | Replace `/model` favorites exactly |
| `clr config` | Replace and validate the stored OpenRouter key |
| `clr config --zai-key` | Store a Z.ai Coding Plan key |
| `clr config --wafer-key` | Store a Wafer Serverless key |
| `clr config --fireworks-key` | Store a Fireworks AI key |
| `clr config --inco-key` | Store an Inco AI key |
| `clr config --cursor-key` | Store a Cursor Cloud Agents key |
| `clr metrics [--days N] [--json] [--histogram] [--model S]` | Token usage, cache, and speed metrics |
| `clr doctor` | Check the service, favorites, and native Claude login |
| `clr claude [ARGS...]` | Compatibility alias for `claude [ARGS...]` |
| `clr update` | Update the installation from GitHub and report the version change |
| `clr reset` | Restore the original Claude settings and delete tool data |
| `clr uninstall` | Reset the integration and remove the installation |

## Credits

Forked from [xhluca/claude-openrouter](https://github.com/xhluca/claude-openrouter),
which pioneered the loopback hybrid-router pattern for Claude Code. This fork
adds the metrics layer, the `[1m]` context-budget handling, deferred tool
loading, the schema repairs, and the Wafer, Fireworks, Inco, and Cursor routes.
