"""Update the installed CLI with the package manager that owns it."""

from __future__ import annotations

import json
import os
import re
import shutil
import site
import subprocess
import sys
from importlib import metadata
from pathlib import Path

PACKAGE_NAME = "claude-router"
PACKAGE_SOURCE = "git+https://github.com/AndresPrez/claude-router"


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.absolute().relative_to(directory.expanduser().absolute())
    except ValueError:
        return False
    return True


def _command_stdout(command: list[str]) -> str | None:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    return output or None


def _fallback_tool_dir() -> Path:
    configured = os.environ.get("CLAUDE_ROUTER_TOOL_DIR")
    if configured:
        return Path(configured)
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return data_home / PACKAGE_NAME / "tool"


def _is_editable_install() -> bool:
    try:
        direct_url = metadata.distribution(PACKAGE_NAME).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return False
    if not direct_url:
        return False
    try:
        document = json.loads(direct_url)
    except (TypeError, json.JSONDecodeError):
        return False
    return bool(document.get("dir_info", {}).get("editable"))


def _pip_command(*, user: bool = False) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--upgrade",
    ]
    if user:
        command.append("--user")
    command.append(PACKAGE_SOURCE)
    return command


def _upgrade_command() -> list[str]:
    """Return an in-place upgrade command for the environment running clr."""
    environment = Path(sys.prefix)

    uv = shutil.which("uv")
    if uv:
        uv_tools = _command_stdout([uv, "tool", "dir"])
        if uv_tools and _inside(environment, Path(uv_tools)):
            return [
                uv,
                "tool",
                "install",
                "--force",
                "--link-mode",
                "copy",
                "--refresh-package",
                PACKAGE_NAME,
                PACKAGE_SOURCE,
            ]

    if _inside(environment, _fallback_tool_dir()):
        return _pip_command()

    pipx = shutil.which("pipx")
    if pipx:
        pipx_venvs = _command_stdout([pipx, "environment", "--value", "PIPX_LOCAL_VENVS"])
        if pipx_venvs and _inside(environment, Path(pipx_venvs)):
            return [pipx, "upgrade", PACKAGE_NAME]

    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        if _is_editable_install():
            raise RuntimeError(
                "this is an editable development install; update it with its source checkout"
            )
        return _pip_command()

    try:
        package_root = Path(metadata.distribution(PACKAGE_NAME).locate_file(""))
        user_site = Path(site.getusersitepackages())
    except (metadata.PackageNotFoundError, TypeError):
        package_root = Path("/")
        user_site = Path("/dev/null")
    if _inside(package_root, user_site):
        return _pip_command(user=True)

    raise RuntimeError(
        "this install is not managed by uv, pipx, the curl installer, or a private Python "
        "environment; update it with the package manager that installed it"
    )


def _installed_version() -> str:
    scripts = Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")
    candidates = [scripts / "clr", scripts / "claude-router"]
    for name in ("clr", "claude-router"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))

    seen: set[Path] = set()
    for executable in candidates:
        if executable in seen or not executable.is_file():
            continue
        seen.add(executable)
        result = subprocess.run(
            [str(executable), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            continue
        reported = result.stdout.strip()
        version = reported.removeprefix(f"{PACKAGE_NAME} ")
        if version and not re.search(r"\s|[\x00-\x1f\x7f]", version):
            return version
    raise RuntimeError("the updated clr executable could not be verified")


def update_installed_package(previous_version: str) -> None:
    """Upgrade clr in place and report the version transition."""
    print("Checking for the latest Claude Router release…")
    command = _upgrade_command()
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Claude Router update exited with status {result.returncode}")
    installed_version = _installed_version()
    if installed_version == previous_version:
        print(f"Claude Router is already up to date at {installed_version}.")
    else:
        print(f"Updated Claude Router from {previous_version} to {installed_version}.")
