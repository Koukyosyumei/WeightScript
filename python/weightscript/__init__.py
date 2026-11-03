"""Public WeightScript API. Model libraries are imported on first API access."""

from importlib import import_module

from .exceptions import MissingDependencyError, dependency_error

__version__ = "0.2.0"

_EXPORTS = {
    "load_model": ("core", "load_model"),
    "extract": ("core", "extract"),
    "program_forward": ("core", "program_forward"),
    "compile_edits": ("core", "compile_edits"),
    "audit": ("core", "audit"),
    "fingerprint": ("core", "fingerprint"),
    "transaction": ("raw", "transaction"),
    "save_undo": ("checkpoint", "save_undo"),
    "undo": ("checkpoint", "undo"),
}
__all__ = [*_EXPORTS, "MissingDependencyError", "__version__"]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    try:
        value = getattr(import_module(f".{module}", __name__), attribute)
    except ModuleNotFoundError as error:
        raise dependency_error(error) from error
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
