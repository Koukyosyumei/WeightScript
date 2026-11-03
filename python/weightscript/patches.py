"""Small, source-bound checkpoint deltas for durable agent sessions."""

from __future__ import annotations

import copy
import json
import os
import uuid
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .core import audit, fingerprint
from .storage import atomic_json


@torch.no_grad()
def save_delta(before, after, directory: str | Path) -> dict:
    root = Path(directory)
    if root.exists() and any(root.iterdir()):
        raise ValueError("delta directory must be empty")
    root.mkdir(parents=True, exist_ok=True)
    changed = audit(before, after)
    original, updated = before.state_dict(), after.state_dict()
    tensors, entries = {}, []
    for index, row in enumerate(changed["tensors"]):
        name = row["tensor"]
        src, dst = original[name], updated[name]
        flat_src = src.detach().cpu().reshape(-1)
        flat = dst.detach().cpu().reshape(-1)
        unequal = flat_src != flat
        if flat_src.is_floating_point():
            unequal = unequal & ~(torch.isnan(flat_src) & torch.isnan(flat))
        indices = torch.where(unequal)[0]
        if indices.numel() == 0:
            continue
        key = f"tensor_{index}"
        # Sparse indexing costs eight bytes per index, so dense changes should
        # use full tensors. Store only changed tensors, never the whole model.
        sparse = (
            indices.numel() * (8 + dst.element_size())
            < dst.numel() * dst.element_size()
        )
        if sparse:
            tensors[key + "_indices"] = indices.contiguous()
            tensors[key] = flat[indices].contiguous()
        else:
            tensors[key] = dst.detach().cpu().contiguous().clone()
        entries.append(
            {
                "name": name,
                "key": key,
                "shape": list(dst.shape),
                "dtype": str(dst.dtype),
                "sparse": sparse,
            }
        )
    blob = root / "delta.safetensors"
    if tensors:
        temporary = root / f".delta.{uuid.uuid4().hex}.safetensors"
        try:
            save_file(tensors, str(temporary))
            fd = os.open(temporary, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(temporary, blob)
        finally:
            temporary.unlink(missing_ok=True)
    manifest = {
        "format": "weightscript.delta.v1",
        "source_sha256": fingerprint(before),
        "result_sha256": fingerprint(after),
        "tensors": entries,
        "audit": changed,
    }
    atomic_json(root / "manifest.json", manifest)
    return manifest


@torch.no_grad()
def apply_delta(model, directory: str | Path):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("format") != "weightscript.delta.v1" or manifest[
        "source_sha256"
    ] != fingerprint(model):
        raise ValueError("delta does not match the source checkpoint")
    blob = root / "delta.safetensors"
    if manifest["tensors"] and not blob.is_file():
        raise ValueError("delta tensors are missing; refusing partial checkpoint")
    if not manifest["tensors"] and blob.exists():
        raise ValueError("delta tensor inventory mismatch")
    tensors = load_file(str(blob)) if manifest["tensors"] else {}
    expected = {entry["key"] for entry in manifest["tensors"]}
    expected |= {
        entry["key"] + "_indices" for entry in manifest["tensors"] if entry["sparse"]
    }
    if set(tensors) != expected:
        raise ValueError("delta tensor inventory mismatch")
    result = copy.deepcopy(model).eval()
    state = result.state_dict()
    seen = set()
    for entry in manifest["tensors"]:
        name, key = entry["name"], entry["key"]
        if name in seen or name not in state:
            raise ValueError("invalid delta parameter")
        seen.add(name)
        target, value = state[name], tensors[key]
        if (
            list(target.shape) != entry["shape"]
            or str(target.dtype) != entry["dtype"]
            or value.dtype != target.dtype
        ):
            raise ValueError("delta shape/dtype mismatch")
        if entry["sparse"]:
            indices = tensors[key + "_indices"]
            if (
                indices.dtype != torch.int64
                or indices.ndim != 1
                or value.shape != indices.shape
                or indices.numel() == 0
            ):
                raise ValueError("invalid sparse delta")
            if (
                indices.min() < 0
                or indices.max() >= target.numel()
                or len(indices.unique()) != len(indices)
            ):
                raise ValueError("invalid sparse delta indices")
            flat = target.reshape(-1)
            flat[indices.to(target.device)] = value.to(target.device)
        else:
            if target.shape != value.shape:
                raise ValueError("invalid dense delta")
            target.copy_(value)
    if fingerprint(result) != manifest["result_sha256"]:
        raise ValueError("delta result hash mismatch")
    return result
