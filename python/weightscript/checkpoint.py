"""Sparse, hash-checked undo packages for compiled edits."""

import copy
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .core import audit, fingerprint, write_json


def save_undo(before, after, directory):
    output = Path(directory)
    changes = audit(before, after)
    original = before.state_dict()
    names = [r["tensor"] for r in changes["tensors"]]
    # safetensors requires contiguous, non-aliased storage.
    save_file(
        {name: original[name].detach().cpu().contiguous().clone() for name in names},
        str(output / "undo.safetensors"),
    )
    write_json(
        output / "undo.json",
        {
            "before_sha256": fingerprint(before),
            "after_sha256": fingerprint(after),
            "tensors": names,
        },
    )


@torch.no_grad()
def undo(model, directory):
    import json

    source = Path(directory)
    manifest = json.loads((source / "undo.json").read_text())
    if fingerprint(model) != manifest["after_sha256"]:
        raise ValueError("undo package does not match this edited checkpoint")
    original = load_file(str(source / "undo.safetensors"))
    if set(original) != set(manifest["tensors"]):
        raise ValueError("undo tensor list mismatch")
    restored = copy.deepcopy(model).eval()
    state = restored.state_dict()
    for name, tensor in original.items():
        if (
            name not in state
            or tensor.shape != state[name].shape
            or tensor.dtype != state[name].dtype
        ):
            raise ValueError("invalid undo tensor")
        state[name].copy_(tensor)
    if fingerprint(restored) != manifest["before_sha256"]:
        raise ValueError("restored checkpoint hash mismatch")
    return restored
