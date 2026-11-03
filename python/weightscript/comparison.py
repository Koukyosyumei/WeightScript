"""Matched editing comparisons on declared datasets.

Development rows select a candidate. Held-out and collateral rows are scored once
after selection. Identical program and direct edits are an interface control, not
evidence that one interface is better. Small samples are not a superiority result.
"""

from __future__ import annotations

import random
import time

import torch

from .core import apply_raw_edits, audit, compile_edits, extract
from .evaluate import score_answers

UPSTREAM_REVISION = "787398b0f4393a90ad02cd8e71be48d1e85d2b21"
UPSTREAM_REPOSITORY = "https://github.com/lacoco-lab/decompiling_transformers"

TASKS = {
    "bin_majority": {
        "repository": UPSTREAM_REPOSITORY,
        "revision": UPSTREAM_REVISION,
        "source": "crasp/scripts/patching/patching_data.py BinaryMajorityDataset",
        "checkpoint": "bin_majority-@1l1h16d3lr01drop",
        "bos": 2,
        "sep": 3,
        "eos": 4,
        "content": 2,
    },
    "unique_copy": {
        "repository": UPSTREAM_REPOSITORY,
        "revision": UPSTREAM_REVISION,
        "source": "crasp/scripts/patching/patching_data.py UniqueCopyDataset",
        "checkpoint": "unique_copy-@2l1h64d3lr01drop",
        "bos": 150,
        "sep": 151,
        "eos": 152,
        "content": 150,
    },
    "sort": {
        "repository": UPSTREAM_REPOSITORY,
        "revision": UPSTREAM_REVISION,
        "source": "crasp/scripts/patching/patching_data.py SortDataset",
        "checkpoint": "sort-@1l1h256d3lr01drop",
        "bos": 150,
        "sep": 151,
        "eos": 152,
        "content": 150,
    },
}


def _disjoint(name, left, right):
    overlap = {tuple(row["input_ids"]) for row in left} & {
        tuple(row["input_ids"]) for row in right
    }
    if overlap:
        raise ValueError(f"{name} inputs overlap")


def majority_rows(seed, lengths, count):
    """Upstream binary-majority prompts. Ties are excluded. Answer follows <sep>."""
    spec = TASKS["bin_majority"]
    rng = random.Random(seed)
    rows, seen = [], set()
    for length in lengths:
        made = 0
        guard = 0
        while made < count:
            guard += 1
            if guard > count * 1000:
                raise ValueError("could not draw enough distinct majority rows")
            zeros = rng.randint(0, length)
            ones = length - zeros
            if zeros == ones:
                continue
            bits = [0] * zeros + [1] * ones
            rng.shuffle(bits)
            prompt = [spec["bos"], *bits, spec["sep"]]
            key = tuple(prompt)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"input_ids": prompt, "targets": [0 if zeros > ones else 1]})
            made += 1
    return rows


def sequence_rows(task, seed, lengths, count):
    """Unique-copy or sort, using the upstream special-token layout."""
    if task not in {"unique_copy", "sort"}:
        raise ValueError("sequence task must be unique_copy or sort")
    spec = TASKS[task]
    rng = random.Random(seed)
    rows, seen = [], set()
    for length in lengths:
        made = 0
        guard = 0
        while made < count:
            guard += 1
            if guard > count * 1000:
                raise ValueError(f"could not draw enough distinct {task} rows")
            tokens = rng.sample(range(spec["content"]), length)
            prompt = [spec["bos"], *tokens, spec["sep"]]
            key = tuple(prompt)
            if key in seen:
                continue
            seen.add(key)
            answer = tokens if task == "unique_copy" else sorted(tokens)
            rows.append({"input_ids": prompt, "targets": [*answer, spec["eos"]]})
            made += 1
    return rows


def task_rows(task, seed, lengths, count):
    if task not in TASKS:
        raise ValueError(f"unsupported comparison task {task!r}")
    if task == "bin_majority":
        return majority_rows(seed, lengths, count)
    return sequence_rows(task, seed, lengths, count)


def _apply(model, proposal):
    kind = proposal["kind"]
    if kind == "none":
        return model
    if kind == "program":
        return compile_edits(model, dict(extract(model), edits=proposal["edits"]))
    if kind == "raw":
        return apply_raw_edits(model, proposal["raw"])
    if kind == "hybrid":
        edited = compile_edits(model, dict(extract(model), edits=proposal["edits"]))
        return apply_raw_edits(edited, proposal["raw"])
    raise ValueError(f"unsupported proposal kind {kind!r}")


def _head_scale(layer, factor):
    return {"op": "head_scale", "layer": layer, "head": 0, "factor": factor}


def _raw_head_scale(model, layer, factor):
    width = model.config.n_embd // model.config.n_head
    return {
        "op": "scale",
        "tensor": f"transformer.h.{layer}.attn.c_proj.weight",
        "rows": list(range(0, width)),
        "factor": factor,
    }


def _direction(model):
    weight = model.transformer.h[0].mlp.c_proj.weight.detach().float()
    _left, _values, right = torch.linalg.svd(weight, full_matrices=False)
    return right[0].tolist()


def proposals_for(model, arm, seed, trials=4):
    """Build a fixed-size candidate list. Unknown arms fail closed."""
    if model.config.model_type != "gpt2":
        raise ValueError("comparison edits currently support GPT-2 checkpoints only")
    if trials < 1:
        raise ValueError("trial count must be positive")
    if arm == "untouched":
        return [{"kind": "none"}]
    factors = [0.5, 0.75, 1.25, 1.5, 0.25, 2.0]
    layers = list(range(model.config.n_layer))
    grid = [(layer, factor) for factor in factors for layer in layers]
    if len(grid) < trials:
        raise ValueError("model does not have enough distinct scaling edits")
    chosen = grid[:trials]
    if arm == "scaling":
        return [
            {"kind": "program", "edits": [_head_scale(layer, factor)]}
            for layer, factor in chosen
        ]
    if arm == "direct":
        return [
            {"kind": "raw", "raw": [_raw_head_scale(model, layer, factor)]}
            for layer, factor in chosen
        ]
    if arm == "program":
        return [
            {"kind": "program", "edits": [_head_scale(layer, factor)]}
            for layer, factor in chosen
        ]
    if arm == "hybrid":
        raw = {
            "op": "scale",
            "tensor": "transformer.h.0.mlp.c_fc.weight",
            "rows": [0],
            "factor": 0.9,
        }
        return [
            {
                "kind": "hybrid",
                "edits": [_head_scale(layer, factor)],
                "raw": [raw],
            }
            for layer, factor in chosen
        ]
    if arm == "direction":
        strengths = [0.25, 0.5, 1.0, 2.0][:trials]
        if len(strengths) < trials:
            raise ValueError("direction arm needs at most four strengths")
        vector = _direction(model)
        neurons = list(range(model.transformer.h[0].mlp.c_proj.weight.shape[0]))
        return [
            {
                "kind": "program",
                "edits": [
                    {
                        "op": "remove_direction",
                        "layer": 0,
                        "neurons": neurons,
                        "direction": vector,
                        "strength": strength,
                    }
                ],
            }
            for strength in strengths
        ]
    if arm == "random":
        rng = random.Random(seed)
        width = model.config.n_embd
        result = []
        for _ in range(trials):
            layer = rng.randrange(model.config.n_layer)
            row = rng.randrange(width)
            factor = rng.choice([0.25, 0.5, 1.5, 2.0])
            result.append(
                {
                    "kind": "raw",
                    "raw": [
                        {
                            "op": "scale",
                            "tensor": f"transformer.h.{layer}.attn.c_proj.weight",
                            "rows": [row],
                            "factor": factor,
                        }
                    ],
                }
            )
        return result
    raise ValueError(f"unsupported comparison arm {arm!r}")


def token_windows(token_ids, prompt_length, count):
    """Non-overlapping next-token windows. The target is the token after the prompt."""
    if prompt_length < 1 or count < 1:
        raise ValueError("window size and count must be positive")
    rows = []
    width = prompt_length + 1
    for start in range(0, len(token_ids) - width + 1, width):
        chunk = list(token_ids[start : start + width])
        rows.append({"input_ids": chunk[:-1], "targets": [chunk[-1]]})
        if len(rows) == count:
            return rows
    raise ValueError(f"passage produced {len(rows)} windows, {count} were required")


ARMS = (
    "untouched",
    "random",
    "scaling",
    "direction",
    "direct",
    "program",
    "hybrid",
)


def compare(model, dev, heldout, collateral, seed, trials=4, arms=ARMS):
    """Select on dev only. Score held-out and collateral after selection."""
    if not dev or not heldout or not collateral:
        raise ValueError("dev, held-out and collateral sets are required")
    _disjoint("development and held-out", dev, heldout)
    _disjoint("development and collateral", dev, collateral)
    _disjoint("held-out and collateral", heldout, collateral)
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        raise ValueError(f"unsupported comparison arms: {unknown}")
    started = time.monotonic()
    methods = []
    for arm in arms:
        proposals = proposals_for(model, arm, seed, trials if arm != "untouched" else 1)
        best = None
        trials_report = []
        dev_calls = 0
        arm_started = time.monotonic()
        for index, proposal in enumerate(proposals):
            candidate = _apply(model, proposal)
            metrics = score_answers(candidate, dev)
            dev_calls += metrics["forward_calls"]
            changed = audit(model, candidate)["changed_scalars"]
            trials_report.append(
                {
                    "candidate": index,
                    "dev_accuracy": metrics["accuracy"],
                    "dev_nll": metrics["target_nll"],
                    "changed_scalars": changed,
                }
            )
            key = (metrics["accuracy"], -metrics["target_nll"])
            if best is None or key > best[0]:
                best = (key, candidate, index, changed, metrics)
        _key, candidate, index, changed, dev_metrics = best
        methods.append(
            {
                "method": arm,
                "selected_candidate": index,
                "candidates_evaluated": len(proposals),
                "development_forward_calls": dev_calls,
                "development": {
                    "accuracy": dev_metrics["accuracy"],
                    "target_nll": dev_metrics["target_nll"],
                    "accuracy_wilson_95": dev_metrics["accuracy_wilson_95"],
                },
                "heldout": score_answers(candidate, heldout, reference=model),
                "collateral": score_answers(candidate, collateral, reference=model),
                "changed_scalars": changed,
                "seconds": time.monotonic() - arm_started,
                "solver_seconds": 0.0,
                "compiler_probe_forwards": 0,
                "matched_interface_control": arm in {"direct", "program", "scaling"},
                "selection_domain": "development",
            }
        )
    return {
        "seed": seed,
        "trials_per_selective_arm": trials,
        "arms": list(arms),
        "methods": methods,
        "seconds": time.monotonic() - started,
        "claim": (
            "Development selection with one held-out and collateral measurement. "
            "Direct, program and scaling arms use the same head-scale grid. "
            "A small accuracy or NLL difference is not evidence of superiority."
        ),
    }
