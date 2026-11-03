"""Capture executable framework source and dependency versions for experiments."""

import hashlib
import importlib.metadata
from pathlib import Path


def runtime():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    packages = {}
    for name in (
        "weightscript",
        "torch",
        "transformers",
        "numpy",
        "tracr",
        "jax",
        "scipy",
        "z3-solver",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"framework_python_sha256": digest.hexdigest(), "packages": packages}
