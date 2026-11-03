"""Generic shape-preserving tensor transactions, independent of model adapters."""

import copy
import math

import torch


def _storage_key(tensor):
    return (tensor.device, tensor.untyped_storage().data_ptr())


def transaction_plan(model, edits):
    """Report storage costs before execution; sparse mode shares untouched parameters.

    Shared parameters must be treated as read-only by both callers. This is explicit
    structural sharing, not automatic copy-on-write for arbitrary PyTorch writes.
    Buffers are always copied because native forward may update rotary caches.
    """
    if not isinstance(edits, list):
        raise ValueError("edits must be a list")
    state = model.state_dict()
    if any(not isinstance(e, dict) or e.get("tensor") not in state for e in edits):
        raise ValueError("unknown tensor")
    touched = {_storage_key(state[e["tensor"]]) for e in edits}
    buffers = {_storage_key(t) for t in model.buffers()}
    storages = {
        _storage_key(t): t.untyped_storage().nbytes()
        for t in list(state.values()) + list(model.buffers())
    }
    copied = touched | buffers
    return {
        "total_storage_bytes": sum(storages.values()),
        "sparse_copy_bytes": sum(v for k, v in storages.items() if k in copied),
        "shared_storage_bytes": sum(v for k, v in storages.items() if k not in copied),
        "edited_tensors": sorted({e["tensor"] for e in edits}),
    }


def isolated_copy(model, touched=(), share_untouched=False):
    """Copy module structure, preserving exact ties and isolating edited storage."""
    if not share_untouched:
        return copy.deepcopy(model).eval()
    state = model.state_dict()
    keys = {_storage_key(state[name]) for name in touched}
    buffers = list(model.buffers())
    keys.update(_storage_key(t) for t in buffers)
    # Non-identical views on one storage require storage-aware view rebuilding.
    # Reject those rare cases instead of silently breaking their alias contract.
    layouts = {}
    for tensor in list(state.values()) + buffers:
        layouts.setdefault(_storage_key(tensor), set()).add(
            (tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()))
        )
    if any(len(v) > 1 for v in layouts.values()):
        raise ValueError("sparse copy does not support overlapping storage views")
    memo, replacements = {}, {}
    for parameter in model.parameters():
        key = (
            _storage_key(parameter),
            parameter.storage_offset(),
            tuple(parameter.shape),
            tuple(parameter.stride()),
        )
        if key not in replacements:
            data = parameter.detach().clone() if key[0] in keys else parameter.detach()
            replacements[key] = torch.nn.Parameter(
                data, requires_grad=parameter.requires_grad
            )
        memo[id(parameter)] = replacements[key]
    for buffer in buffers:
        key = (
            _storage_key(buffer),
            buffer.storage_offset(),
            tuple(buffer.shape),
            tuple(buffer.stride()),
        )
        if key not in replacements:
            replacements[key] = buffer.detach().clone()
        memo[id(buffer)] = replacements[key]
    return copy.deepcopy(model, memo).eval()


def select(tensor, selectors):
    if not isinstance(selectors, list) or len(selectors) != tensor.ndim:
        raise ValueError("one selector per tensor axis is required")
    axes = []
    for size, s in zip(tensor.shape, selectors):
        if s is None:
            axes.append(torch.arange(size, device=tensor.device))
        elif isinstance(s, dict) and set(s) <= {"start", "stop", "step"}:
            start = s.get("start", 0)
            stop = s.get("stop", size)
            step = s.get("step", 1)
            if (
                any(type(v) is not int for v in (start, stop, step))
                or not 0 <= start < stop <= size
                or step <= 0
            ):
                raise ValueError("invalid tensor slice")
            axes.append(torch.arange(start, stop, step, device=tensor.device))
        elif (
            isinstance(s, list)
            and s
            and all(type(i) is int and 0 <= i < size for i in s)
            and len(s) == len(set(s))
        ):
            axes.append(torch.tensor(s, device=tensor.device))
        else:
            raise ValueError("invalid tensor selector")
    return tuple(torch.meshgrid(*axes, indexing="ij"))


@torch.no_grad()
def transaction(model, edits, *, share_untouched=False):
    """Apply edits atomically; source and tied aliases are never written.

    Default output is fully isolated. Opt-in sharing saves untouched parameter
    storage, with the read-only ownership requirements in transaction_plan.
    """
    transaction_plan(model, edits)
    result = isolated_copy(model, [e["tensor"] for e in edits], share_untouched)
    # Use state_dict so either tied alias can be addressed; edits accumulate on shared storage.
    state = result.state_dict()
    if not isinstance(edits, list):
        raise ValueError("edits must be a list")
    for edit in edits:
        if not isinstance(edit, dict) or edit.get("tensor") not in state:
            raise ValueError("unknown tensor")
        tensor = state[edit["tensor"]]
        if not tensor.is_floating_point() or tensor.dtype not in {
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
        }:
            raise ValueError("raw edits require real floating tensors")
        op = edit.get("op")
        if op in {"set", "add", "scale"}:
            if set(edit) != {"op", "tensor", "select", "value"}:
                raise ValueError("invalid raw transaction fields")
            indices = select(tensor, edit["select"])
            original = tensor[indices]
            value = torch.as_tensor(
                edit["value"], dtype=tensor.dtype, device=tensor.device
            )
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite edit")
            if value.ndim and value.shape != original.shape:
                raise ValueError("edit value shape mismatch")
            updated = (
                value
                if op == "set"
                else original + value
                if op == "add"
                else original * value
            )
            tensor[indices] = updated
        elif op == "low_rank_add":
            if (
                set(edit) != {"op", "tensor", "left", "right", "factor"}
                or tensor.ndim != 2
            ):
                raise ValueError("invalid low-rank update")
            left = tensor.new_tensor(edit["left"])
            right = tensor.new_tensor(edit["right"])
            factor = edit["factor"]
            if (
                left.ndim != 2
                or right.ndim != 2
                or left.shape[0] != tensor.shape[0]
                or right.shape[1] != tensor.shape[1]
                or left.shape[1] != right.shape[0]
                or not torch.isfinite(left).all()
                or not torch.isfinite(right).all()
            ):
                raise ValueError("low-rank factors do not fit tensor")
            if type(factor) not in (int, float) or not math.isfinite(factor):
                raise ValueError("invalid update factor")
            tensor.add_(factor * (left @ right))
        else:
            raise ValueError("unknown transaction operation")
    if any(not torch.isfinite(p).all() for p in result.state_dict().values()):
        raise ValueError("transaction produced nonfinite weights")
    return result
