#!/bin/sh
# POSIX user-level installer for Claude Router.

set -eu

package_name="claude-router"
package_source="git+https://github.com/AndresPrez/claude-router"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Install Claude Router for the current user and run its guided setup.

Usage:
  curl -LsSf https://andresperez.github.io/claude-router/install.sh | sh
  sh install.sh [options]

Options:
  --install-only        Install the CLI without asking for a key or models.
  --skip-claude-install Fail instead of installing Claude Code when it is missing.
  -h, --help            Show this help.

The installer never accepts an API key as a command-line argument. Guided setup
uses a masked terminal prompt and stores the key in a mode-0600 credential file.
EOF
}

install_only=0
skip_claude_install=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --install-only) install_only=1; shift ;;
    --skip-claude-install) skip_claude_install=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (run with --help)" ;;
  esac
done

script_dir=""
case "$0" in
  */*) script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" 2>/dev/null && pwd -P) || script_dir='' ;;
  *)
    if [ -f "./$0" ]; then
      script_dir=$(pwd -P)
    fi
    ;;
esac

local_source=""
if [ -n "$script_dir" ] && [ -f "$script_dir/pyproject.toml" ]; then
  local_source="$script_dir"
fi

if ! command -v claude >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/claude" ]; then
  [ "$skip_claude_install" -eq 0 ] || die "Claude Code is not installed"
  command -v curl >/dev/null 2>&1 || die "curl is required to install Claude Code"
  printf 'Installing Claude Code from its official installer...\n'
  curl -fsSL https://claude.ai/install.sh | bash
fi

printf 'Installing %s from GitHub for user %s...\n' "$package_name" "$(id -un)"
installed_command=""

if command -v uv >/dev/null 2>&1; then
  if [ -n "$local_source" ]; then
    uv tool install --force --link-mode copy "$local_source"
  elif ! uv tool install --force --link-mode copy --refresh-package "$package_name" \
    "$package_source"; then
    die "installing $package_name from GitHub failed"
  fi
  uv_bin_dir="${UV_TOOL_BIN_DIR:-$HOME/.local/bin}"
  if [ -x "$uv_bin_dir/claude-router" ]; then
    installed_command="$uv_bin_dir/claude-router"
  elif command -v claude-router >/dev/null 2>&1; then
    installed_command=$(command -v claude-router)
  fi
else
  python="${PYTHON:-python3}"
  command -v "$python" >/dev/null 2>&1 || die "Python 3.10+ or uv is required"
  "$python" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' || \
    die "Python 3.10 or newer is required"
  command -v git >/dev/null 2>&1 || \
    die "git is required to install from GitHub; install git or use uv"
  data_home="${XDG_DATA_HOME:-$HOME/.local/share}"
  bin_dir="${XDG_BIN_HOME:-$HOME/.local/bin}"
  install_dir="${CLAUDE_ROUTER_TOOL_DIR:-$data_home/claude-router/tool}"
  mkdir -p -- "$install_dir" "$bin_dir"
  for destination in "$bin_dir/claude-router" "$bin_dir/clr"; do
    if [ -e "$destination" ] && [ ! -L "$destination" ]; then
      die "refusing to replace existing file: $destination"
    fi
  done
  "$python" -m venv "$install_dir"
  if [ -n "$local_source" ]; then
    "$install_dir/bin/python" -m pip install --disable-pip-version-check --force-reinstall "$local_source"
  elif ! "$install_dir/bin/python" -m pip install --disable-pip-version-check \
    --force-reinstall "$package_source"; then
    die "installing $package_name from GitHub failed"
  fi
  ln -sfn -- "$install_dir/bin/claude-router" "$bin_dir/claude-router"
  ln -sfn -- "$install_dir/bin/clr" "$bin_dir/clr"
  installed_command="$bin_dir/claude-router"
fi

[ -n "$installed_command" ] && [ -x "$installed_command" ] || \
  die "installation completed but claude-router was not found"
printf 'Installed commands: %s and clr\n' "$installed_command"

if [ "$install_only" -eq 1 ]; then
  printf 'Installation complete. Run: %s setup\n' "$installed_command"
  exit 0
fi

if [ -r /dev/tty ]; then
  "$installed_command" setup </dev/tty
else
  die "interactive setup needs a terminal; rerun claude-router setup directly"
fi

printf '\nClaude Router is ready. Run: claude\n'
