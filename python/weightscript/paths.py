"""User-configurable application directories; independent of any checkout."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def state_directory() -> Path:
    """Return the durable user state directory without creating it."""
    if override := os.environ.get("WEIGHTSCRIPT_STATE_DIR"):
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return base / "weightscript"


def cache_directory() -> Path:
    """Return the disposable model/backend cache directory."""
    if override := os.environ.get("WEIGHTSCRIPT_CACHE_DIR"):
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = (
            Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
            / "Cache"
        )
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "weightscript"
