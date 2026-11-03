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
def score_answers(model, rows, batch_size=32, reference=None):
    """Teacher-forced score of every answer token. Prompts stay in input_ids.

    `targets` is the answer sequence. One-token tasks use a one-element list.
    Held-out isolation is the caller's job; this function only scores the rows
    it is given.
    """
    if not rows or any(not row.get("targets") for row in rows):
        raise ValueError("answer scoring needs nonempty targets")
    prepared = []
    for row in rows:
        prompt, targets = row["input_ids"], row["targets"]
        if (
            not isinstance(prompt, list)
            or not isinstance(targets, list)
            or not prompt
            or not targets
            or any(type(token) is not int for token in prompt + targets)
        ):
            raise ValueError("prompts and targets must be nonempty integer lists")
        prepared.append((prompt + targets[:-1], targets, len(prompt)))
    correct = total = calls = changed = positions = 0
    nll = nll_squared = kl = 0.0
    groups = defaultdict(list)
    for item in prepared:
        groups[len(item[0])].append(item)
    for length in sorted(groups):
        group = groups[length]
        for start in range(0, len(group), batch_size):
            chunk = group[start : start + batch_size]
            ids = torch.tensor([item[0] for item in chunk], dtype=torch.long)
            logits = model(ids, use_cache=False).logits
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite model logits")
            calls += 1
            reference_logits = (
                reference(ids, use_cache=False).logits
                if reference is not None
                else None
            )
            for index, (_, targets, prompt_length) in enumerate(chunk):
                for offset, target in enumerate(targets):
                    position = prompt_length - 1 + offset
                    predicted = logits[index, position]
                    if not 0 <= target < predicted.shape[-1]:
                        raise ValueError("target outside vocabulary")
                    correct += int(int(predicted.argmax()) == target)
                    loss = float(-predicted.log_softmax(-1)[target])
                    nll += loss
                    nll_squared += loss * loss
                    total += 1
                    if reference_logits is not None:
                        base = reference_logits[index, position]
                        changed += int(int(predicted.argmax()) != int(base.argmax()))
                        positions += 1
                        kl += float(
                            F.kl_div(
                                predicted.log_softmax(-1),
                                base.softmax(-1),
                                reduction="sum",
                            )
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


def count_forwards(rows, batch_size, objective="next_token"):
    """Forward groups charged for one scoring pass.

    Next-token scoring groups prompt lengths. Sequence scoring groups the
    teacher-forced length, `len(input_ids) + len(targets) - 1`.
    """
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    if objective == "next_token":
        return sum(1 for _ in batches(rows, batch_size))
    if objective != "sequence":
        raise ValueError(f"unsupported objective {objective!r}")
    if not rows:
        raise ValueError("sequence scoring needs nonempty targets")
    groups = defaultdict(int)
    for row in rows:
        prompt, targets = row.get("input_ids"), row.get("targets")
        if (
            not isinstance(prompt, list)
            or not isinstance(targets, list)
            or not prompt
            or not targets
        ):
            raise ValueError("sequence scoring needs nonempty targets")
        groups[len(prompt) + len(targets) - 1] += 1
    return sum((count + batch_size - 1) // batch_size for count in groups.values())


@torch.inference_mode()
def score_objective(model, rows, batch_size=32, reference=None, objective="next_token"):
    """Score `next_token` at the last prompt position, or every id in `targets`."""
    if objective == "next_token":
        if not rows or any(
            "target" not in row or type(row["target"]) is not int for row in rows
        ):
            raise ValueError("next-token scoring needs an integer target on every row")
        metrics = score(model, rows, batch_size, reference)
        lengths = [1] * len(rows)
        full_span = False
    elif objective == "sequence":
        metrics = score_answers(model, rows, batch_size, reference)
        lengths = [len(row["targets"]) for row in rows]
        full_span = True
    else:
        raise ValueError(f"unsupported objective {objective!r}")
    metrics["objective"] = objective
    metrics["min_answer_tokens"] = min(lengths) if lengths else 0
    metrics["max_answer_tokens"] = max(lengths) if lengths else 0
    metrics["full_span"] = full_span
    return metrics


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
