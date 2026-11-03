"""Portable directories for dataset caches and result files.

No path in this module is a machine. Set ``WEIGHTSCRIPT_CACHE_DIR`` for a
dataset cache, ``WEIGHTSCRIPT_RUNS_DIR`` for result files, and ``TMPDIR`` for
temporary downloads. Unset variables do not create directories and do not
replace Hugging Face's own cache.
"""

from __future__ import annotations

import os
from pathlib import Path


def _explicit(value, label: str) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        raise ValueError(f"{label} is empty")
    return Path(text).expanduser()


def dataset_cache_dir(explicit: str | None = None) -> str | None:
    """Cache directory to pass into ``datasets.load_dataset``.

    An explicit argument wins. ``HF_DATASETS_CACHE`` is left for the datasets
    library to read. ``WEIGHTSCRIPT_CACHE_DIR`` is used only when neither of
    those is set, and then only as ``<dir>/datasets``.
    """

    chosen = _explicit(explicit, "cache_dir")
    if chosen is not None:
        return str(chosen)
    if os.environ.get("HF_DATASETS_CACHE"):
        return None
    override = os.environ.get("WEIGHTSCRIPT_CACHE_DIR")
    if override and override.strip():
        return str(Path(override).expanduser() / "datasets")
    return None


def runs_dir(explicit: str | None = None) -> Path | None:
    """Directory for result files. None when the caller has not chosen one."""

    chosen = _explicit(explicit, "runs_dir")
    if chosen is not None:
        return chosen
    override = os.environ.get("WEIGHTSCRIPT_RUNS_DIR")
    if override and override.strip():
        return Path(override).expanduser()
    return None
