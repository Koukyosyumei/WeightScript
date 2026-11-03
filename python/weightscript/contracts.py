"""Explicit architecture contracts.

Runtime adapters must reject anything outside these declarations. This module has
no model-library dependency so help and contract checks can run without them.
"""

from __future__ import annotations

_FLOAT = ("float32", "float16", "bfloat16")
_DEVICES = ("cpu", "cuda")
_ROPE = ("default", "linear", "dynamic", "llama3")

CONTRACTS = {
    "gpt2": {
        "model_type": "gpt2",
        "class_name": "GPT2LMHeadModel",
        "precision": ("float32",),
        "device": ("cpu",),
        "rope": (),
        "attention": ("causal",),
        "grouped_query": False,
        "sliding_window": False,
        "notes": "Learned absolute positions. No RoPE and no sliding window.",
    },
    "llama": {
        "model_type": "llama",
        "class_name": "LlamaForCausalLM",
        "precision": _FLOAT,
        "device": _DEVICES,
        "rope": _ROPE,
        "attention": ("causal", "grouped_query"),
        "grouped_query": True,
        "sliding_window": False,
        "notes": "Windowed Llama configs are rejected because the pinned native model does not implement them.",
    },
    "qwen2": {
        "model_type": "qwen2",
        "class_name": "Qwen2ForCausalLM",
        "precision": _FLOAT,
        "device": _DEVICES,
        "rope": _ROPE,
        "attention": ("causal", "grouped_query", "sliding_window"),
        "grouped_query": True,
        "sliding_window": True,
        "notes": "Mixed full and sliding layers follow config.layer_types.",
    },
    "mistral": {
        "model_type": "mistral",
        "class_name": "MistralForCausalLM",
        "precision": _FLOAT,
        "device": _DEVICES,
        "rope": _ROPE,
        "attention": ("causal", "grouped_query", "sliding_window"),
        "grouped_query": True,
        "sliding_window": True,
        "notes": "Dynamic NTK needs an explicit numeric head_dim on Transformers 4.57.1.",
    },
}


def backend_contract(model_type: str) -> dict:
    """Return a copy of the support contract, or fail closed for unknown families."""
    if model_type not in CONTRACTS:
        raise ValueError(
            f"unsupported architecture {model_type!r}; no semantic contract and no native-forward fallback"
        )
    contract = dict(CONTRACTS[model_type])
    contract["precision"] = tuple(contract["precision"])
    contract["device"] = tuple(contract["device"])
    contract["rope"] = tuple(contract["rope"])
    contract["attention"] = tuple(contract["attention"])
    return contract
