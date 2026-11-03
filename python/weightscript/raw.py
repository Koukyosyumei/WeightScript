"""Generic shape-preserving tensor transactions, independent of model adapters."""

import copy
import math

import torch


def select(tensor, selectors):
    if len(selectors) != tensor.ndim:
        raise ValueError("one selector per tensor axis is required")
    axes = []
    for size, s in zip(tensor.shape, selectors):
        if s is None:
            axes.append(torch.arange(size))
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
            axes.append(torch.arange(start, stop, step))
        elif (
            isinstance(s, list)
            and s
            and len(s) == len(set(s))
            and all(type(i) is int and 0 <= i < size for i in s)
        ):
            axes.append(torch.tensor(s))
        else:
            raise ValueError("invalid tensor selector")
    return tuple(torch.meshgrid(*axes, indexing="ij"))


@torch.no_grad()
def transaction(model, edits):
    result = copy.deepcopy(model).eval()
    # Use state_dict so either tied alias can be addressed; edits accumulate on shared storage.
    state = result.state_dict()
    if not isinstance(edits, list):
        raise ValueError("edits must be a list")
    for edit in edits:
        if not isinstance(edit, dict) or edit.get("tensor") not in state:
            raise ValueError("unknown tensor")
        tensor = state[edit["tensor"]]
        op = edit.get("op")
        if op in {"set", "add", "scale"}:
            if set(edit) != {"op", "tensor", "select", "value"}:
                raise ValueError("invalid raw transaction fields")
            indices = select(tensor, edit["select"])
            original = tensor[indices]
            value = torch.as_tensor(edit["value"], dtype=tensor.dtype)
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
            ):
                raise ValueError("low-rank factors do not fit tensor")
            if type(factor) not in (int, float) or not math.isfinite(factor):
                raise ValueError("invalid update factor")
            tensor.add_(factor * (left @ right))
        else:
            raise ValueError("unknown transaction operation")
    if any(not torch.isfinite(p).all() for p in result.parameters()):
        raise ValueError("transaction produced nonfinite weights")
    return result
