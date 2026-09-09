<h1 align="center">Claude Router</h1>

<p align="center"><strong>Use Z.ai GLM and OpenRouter models inside Claude Code next to your native Claude login.</strong></p>

Claude Router is a small, dependency-free CLI and loopback router. Your native
Claude login keeps working for Claude models, and favorites from the Z.ai
Coding Plan or OpenRouter appear beside them in Claude Code's native `/model`
picker. After setup, run the normal `claude` command.

## Install

```bash
uv tool install git+https://github.com/AndresPrez/claude-router
```

Or with curl:

```bash
curl -LsSf https://andresperez.github.io/claude-router/install.sh | sh
```

## Quick start

Z.ai Coding Plan only (no OpenRouter account needed):

```bash
clr setup --no-openrouter
```

With an OpenRouter key:

```bash
clr setup
```

Then run `claude` and switch models with `/model`. Built-in Claude models keep
using your native login; rows labeled `· Z.ai` bill to your Coding Plan quota
and rows labeled `· OpenRouter` use your stored OpenRouter key.

## Commands

| Command | Purpose |
| --- | --- |
| `clr index` | Fetch and cache the current OpenRouter model catalog |
| `clr fetch` | Alias for `index` |
| `clr search QUERY...` | Refresh, then search names, IDs, and descriptions |
| `clr search QUERY... --tools` | Show only models advertising tool calling |
| `clr check MODEL [--yes]` | Confirm and run one billable Claude Code tool round-trip |
| `clr setup` | Run the install-time key and model setup again |
| `clr select [MODEL]` | Replace `/model` favorites exactly |
| `clr config` | Replace and validate the stored OpenRouter key |
| `clr config --zai-key` | Store a Z.ai Coding Plan key |
| `clr doctor` | Check the service, favorites, and native Claude login |
| `clr claude [ARGS...]` | Compatibility alias for `claude [ARGS...]` |
| `clr update` | Update the installation from GitHub and report the version change |
| `clr reset` | Restore the original Claude settings and delete tool data |
| `clr uninstall` | Reset the integration and remove the installation |

## Credits

Claude Router is a fork of
[claude-openrouter](https://github.com/xhluca/claude-openrouter) by xhluca,
MIT licensed.
