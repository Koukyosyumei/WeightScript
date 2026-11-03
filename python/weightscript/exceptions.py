"""Public exceptions for dependency diagnostics; core validation uses ValueError."""


class MissingDependencyError(ImportError):
    """An operation needs an unavailable runtime or research dependency."""


_RESEARCH = {
    "jax",
    "jaxlib",
    "haiku",
    "chex",
    "einops",
    "networkx",
    "matplotlib",
    "accelerate",
    "click",
    "z3",
    "scipy",
}


def dependency_error(error):
    if isinstance(error, MissingDependencyError):
        return error
    name = (error.name or "unknown").split(".")[0]
    if name == "tracr":
        hint = "install Tracr separately; see docs/installation.md"
    elif name == "decompiling_transformers":
        hint = "provide the upstream checkout with --checkout; see docs/installation.md"
    elif name in _RESEARCH:
        hint = 'install the research extra: pip install "weightscript[research]"'
    elif name in {"torch", "transformers", "safetensors", "numpy"}:
        hint = "install the runtime dependencies: pip install weightscript"
    else:
        hint = "check the dependency installation and original module name"
    return MissingDependencyError(f"missing dependency '{name}'; {hint}")
