"""Budgeted, reproducible evaluation; held-out data never selects a candidate."""

import json
import hashlib
import math
import platform
import transformers
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .core import (
    audit,
    apply_raw_edits,
    compile_edits,
    extract,
    program_forward,
    validate_program,
)


def dataset(path):
    rows = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError("empty dataset")
    for row in rows:
        if (
            not isinstance(row.get("input_ids"), list)
            or not row["input_ids"]
            or any(type(t) is not int for t in row["input_ids"])
        ):
            raise ValueError("each example needs nonempty integer input_ids")
        if "target" in row and type(row["target"]) is not int:
            raise ValueError("target must be a token id")
    return rows


def batches(rows, batch_size):
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    groups = defaultdict(list)
    for row in rows:
        groups[len(row["input_ids"])].append(row)
    for length in sorted(groups):
        group = groups[length]
        for i in range(0, len(group), batch_size):
            chunk = group[i : i + batch_size]
            yield chunk, torch.tensor([r["input_ids"] for r in chunk], dtype=torch.long)


@torch.inference_mode()
def score(model, rows, batch_size=32, reference=None):
    correct = total = calls = changed = positions = 0
    nll = kl = nll_squared = 0.0
    if not rows:
        raise ValueError("empty evaluation domain")
    for chunk, ids in batches(rows, batch_size):
        logits = model(ids, use_cache=False).logits
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite model logits")
        calls += 1
        for i, row in enumerate(chunk):
            if "target" in row:
                target = row["target"]
                if not 0 <= target < logits.shape[-1]:
                    raise ValueError("target outside vocabulary")
                correct += int(logits[i, -1].argmax() == target)
                loss = float(-logits[i, -1].log_softmax(-1)[target])
                nll += loss
                nll_squared += loss * loss
                total += 1
        if reference is not None:
            base = reference(ids, use_cache=False).logits
            changed += int((logits.argmax(-1) != base.argmax(-1)).sum())
            positions += ids.numel()
            kl += float(
                F.kl_div(logits.log_softmax(-1), base.softmax(-1), reduction="sum")
            )
    interval = None
    if total:
        z = 1.95996398454
        p = correct / total
        center = (p + z * z / (2 * total)) / (1 + z * z / total)
        radius = (
            z
            * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
            / (1 + z * z / total)
        )
        interval = [max(0, center - radius), min(1, center + radius)]
    return {
        "examples": len(rows),
        "labeled": total,
        "accuracy_wilson_95": interval,
        "target_nll_standard_error": math.sqrt(
            max(0, nll_squared - nll * nll / total) / (total - 1) / total
        )
        if total > 1
        else None,
        "accuracy": correct / total if total else None,
        "target_nll": nll / total if total else None,
        "forward_calls": calls,
        "reference_forward_calls": calls if reference is not None else 0,
        "token_disagreement": changed / positions if positions else None,
        "reference_kl_per_token": kl / positions if positions else None,
    }


@torch.inference_mode()
def equivalence(model, program, rows, batch_size=32, atol=2e-5, rtol=2e-5):
    if not rows:
        raise ValueError("empty verification domain")
    validate_program(model, program)
    edited = compile_edits(model, program)
    base = extract(model)
    max_original_error = max_edited_error = 0.0
    passed = True
    calls = 0
    for _, ids in batches(rows, batch_size):
        original_native = model(ids, use_cache=False).logits
        original_ir = program_forward(model, base, ids, validated=True)
        edited_native = edited(ids, use_cache=False).logits
        edited_ir = program_forward(model, program, ids, validated=True)
        max_original_error = max(
            max_original_error, float((original_native - original_ir).abs().max())
        )
        max_edited_error = max(
            max_edited_error, float((edited_native - edited_ir).abs().max())
        )
        passed &= bool(
            torch.allclose(original_native, original_ir, atol=atol, rtol=rtol)
        )
        passed &= bool(torch.allclose(edited_native, edited_ir, atol=atol, rtol=rtol))
        calls += 4
    return {
        "passed": passed,
        "examples": len(rows),
        "forward_calls": calls,
        "max_original_logit_error": max_original_error,
        "max_edited_logit_error": max_edited_error,
        "atol": atol,
        "rtol": rtol,
        "claim": "numerical agreement on supplied examples; not a universal formal proof",
        "audit": audit(model, edited),
    }


def benchmark(model, candidates, dev, heldout, batch_size=32, budget=50):
    if not all("target" in r for r in dev + heldout):
        raise ValueError("benchmark requires labeled examples")
    # Reject exact input overlap before running any candidates.
    if {tuple(r["input_ids"]) for r in dev} & {tuple(r["input_ids"]) for r in heldout}:
        raise ValueError("development and held-out input sets overlap")
    base = extract(model)
    base_dev = score(model, dev, batch_size)
    base_test = score(model, heldout, batch_size)
    results = []
    for method, edits_list in candidates.items():
        if not edits_list:
            raise ValueError(f"no candidates for {method}")
        evaluations = []
        best = None
        used = 0
        per_candidate = sum(1 for _ in batches(dev, batch_size))
        for index, proposal in enumerate(edits_list):
            if used + per_candidate > budget:
                break
            if isinstance(proposal, dict) and set(proposal) == {"raw"}:
                program = {"raw": proposal["raw"]}
                candidate = apply_raw_edits(model, proposal["raw"])
            else:
                edits = (
                    proposal["edits"]
                    if isinstance(proposal, dict) and set(proposal) == {"edits"}
                    else proposal
                )
                program = dict(base, edits=edits)
                candidate = compile_edits(model, program)
            metrics = score(candidate, dev, batch_size)
            used += metrics["forward_calls"]
            evaluations.append(
                {
                    "candidate": index,
                    "dev": metrics,
                    "changed_scalars": audit(model, candidate)["changed_scalars"],
                }
            )
            key = (metrics["accuracy"], -metrics["target_nll"])
            if best is None or key > best[0]:
                best = (key, program, candidate, index)
        if best is None:
            raise ValueError("budget too small for one complete development evaluation")
        _, program, candidate, index = best
        results.append(
            {
                "method": method,
                "selected_candidate": index,
                "development_forward_calls": used,
                "budget": budget,
                "trials": evaluations,
                "program": program,
                "heldout": score(candidate, heldout, batch_size, reference=model),
                "audit": audit(model, candidate),
            }
        )
    from .provenance import runtime

    return {
        "runtime": runtime(),
        "base_sha256": base["base_sha256"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "threads": torch.get_num_threads(),
            "device": "cpu",
            "dtype": "float32",
        },
        "data_sha256": {
            "dev": hashlib.sha256(json.dumps(dev, sort_keys=True).encode()).hexdigest(),
            "heldout": hashlib.sha256(
                json.dumps(heldout, sort_keys=True).encode()
            ).hexdigest(),
        },
        "protocol": "candidate selection on development data; one held-out evaluation per method; equal maximum development forward-call budget",
        "limitations": "candidate proposals supplied externally; not a replication of the blog's LLM experiments; no automatic claim of superiority",
        "base_dev": base_dev,
        "base_heldout": base_test,
        "methods": results,
    }
