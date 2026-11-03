"""Public WeightScript API. Model libraries are imported on first API access."""

from importlib import import_module

from .contracts import backend_contract
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
    "decompile": ("backends.upstream", "decompile"),
    "execute_artifact": ("backends.upstream", "execute_artifact"),
    "decompile_stages": ("backends.upstream_stages", "decompile_stages"),
    "execute_stages": ("backends.upstream_stages", "execute_stages"),
    "match_qk_primitives": ("backends.upstream_primitives", "match_qk_primitives"),
    "disjoint_domains": ("backends.pipeline", "disjoint_domains"),
    "summarize_stages": ("backends.pipeline", "summarize_stages"),
    "decompile_and_edit": ("backends.pipeline", "decompile_and_edit"),
    "validate_rule": ("backends.rules", "validate_rule"),
    "rule_predict": ("backends.rules", "rule_predict"),
    "supervised_pairs": ("backends.rules", "supervised_pairs"),
    "synthesize_rules": ("symbolic", "synthesize_rules"),
    "synthesize_sequence_rules": ("symbolic", "synthesize_sequence_rules"),
    "synthesize_binary": ("symbolic", "synthesize_binary"),
    "lower": ("symbolic", "lower"),
    "compile_rule": ("synthesis", "compile_rule"),
    "cegis": ("synthesis", "cegis"),
    "replay_patch": ("synthesis", "replay_patch"),
}
__all__ = [*_EXPORTS, "MissingDependencyError", "backend_contract", "__version__"]


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
