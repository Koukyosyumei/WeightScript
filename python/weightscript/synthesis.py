"""Bounded symbolic-rule lowering into an existing GPT-2 MLP via linear constraints.

A constrained compiler, not zero-data editing: it observes hidden activations on
an explicit synthesis domain, solves an LP (no gradients), and checks native
float32 execution separately. No claim of unrestricted algorithm equivalence.
"""

import copy
import hashlib
import json

import numpy as np
import torch

from .core import audit, fingerprint
from .evaluate import batches


def rule_predict(rule, ids):
    if rule.get("op") != "count_threshold":
        raise ValueError("constraint backend supports binary count_threshold rules")
    if (
        set(rule) != {"op", "threshold", "invert"}
        or type(rule["invert"]) is not bool
        or not isinstance(rule["threshold"], (int, float))
        or not 0 <= rule["threshold"] <= 1
    ):
        raise ValueError("invalid count rule")
    bits = [t for t in ids if t in (0, 1)]
    if not bits:
        raise ValueError("rule domain needs at least one bit")
    return int((sum(bits) / len(bits) > rule["threshold"]) != rule["invert"])


@torch.inference_mode()
def compile_rule(
    model,
    rule,
    domain,
    verification=None,
    preserve=None,
    margin=0.1,
    coefficient_bound=100,
    timeout=30,
    strict=True,
):
    from scipy.optimize import linprog
    import z3

    if (
        model.config.model_type != "gpt2"
        or not domain
        or margin <= 0
        or coefficient_bound <= 0
    ):
        raise ValueError("invalid constraint compiler model/domain")
    if verification and {tuple(r["input_ids"]) for r in domain} & {
        tuple(r["input_ids"]) for r in verification
    }:
        raise ValueError("synthesis and verification domains overlap")
    if preserve and {tuple(r["input_ids"]) for r in domain} & {
        tuple(r["input_ids"]) for r in preserve
    }:
        raise ValueError("rewrite and preservation domains overlap")
    rows = domain + (preserve or [])
    snapshots = []
    calls = 0
    for chunk, ids in batches(rows, 128):
        hidden = []
        residual = []
        h = model.transformer.h[-1].mlp.c_proj.register_forward_pre_hook(
            lambda _, args: hidden.append(args[0][:, -1].detach().clone())
        )
        r = model.transformer.ln_f.register_forward_pre_hook(
            lambda _, args: residual.append(args[0][:, -1].detach().clone())
        )
        try:
            native = model(ids, use_cache=False).logits[:, -1]
            calls += 1
        finally:
            h.remove()
            r.remove()
        domain_ids = {tuple(row["input_ids"]) for row in domain}
        for j, row in enumerate(chunk):
            target = (
                rule_predict(rule, row["input_ids"])
                if tuple(row["input_ids"]) in domain_ids
                else int(native[j].argmax())
            )
            snapshots.append((hidden[0][j], residual[0][j], target))
    unembed = model.lm_head.weight.detach()
    gamma = model.transformer.ln_f.weight.detach()
    direction = (unembed[1] - unembed[0]) * gamma
    direction -= direction.mean()
    if direction.norm() == 0:
        raise ValueError("binary output contrast has no residual direction")
    direction /= direction.norm()
    feature_count = snapshots[0][0].numel() + 1
    inequalities = []
    upper = []
    for hidden, residual, target in snapshots:
        features = np.append(hidden.double().numpy(), 1.0)
        for other in range(model.config.vocab_size):
            if other == target:
                continue
            diff = ((unembed[target] - unembed[other]) * gamma).double()
            diff -= diff.mean()
            inequalities.append(-float(direction.double() @ diff) * features)
            upper.append(float(residual.double() @ diff) - margin)
    matrix = np.array(inequalities)
    bound = np.array(upper)
    # Positive/negative split minimizes coefficient L1 and often uses few units.
    result = linprog(
        np.ones(2 * feature_count),
        A_ub=np.concatenate((matrix, -matrix), axis=1),
        b_ub=bound,
        bounds=[(0, coefficient_bound)] * (2 * feature_count),
        method="highs",
        options={"time_limit": timeout},
    )
    if not result.success:
        raise ValueError(f"rule does not fit chosen rank-one patch: {result.message}")
    coefficients = result.x[:feature_count] - result.x[feature_count:]
    edited = copy.deepcopy(model).eval()
    co = torch.tensor(coefficients[:-1], dtype=torch.float32)
    edited.transformer.h[-1].mlp.c_proj.weight.add_(co[:, None] * direction[None, :])
    edited.transformer.h[-1].mlp.c_proj.bias.add_(float(coefficients[-1]) * direction)
    # Eliminates the post-normalization affine offset so target comparisons are
    # linear in the pre-normalization residual; this change is audited explicitly.
    edited.transformer.ln_f.bias.zero_()
    if any(not torch.isfinite(p).all() for p in edited.parameters()):
        raise ValueError("nonfinite synthesized weights")
    # Exact rational checker over the LP's recorded real-valued inequalities.
    # This does NOT model the floating-point network or unobserved inputs.
    rational = z3.Solver()
    rational.set(timeout=int(timeout * 1000))
    rational.add(
        z3.Or(
            *[
                z3.Sum(
                    [
                        z3.RealVal(str(a)) * z3.RealVal(str(c))
                        for a, c in zip(row, coefficients)
                    ]
                )
                > z3.RealVal(str(b + 1e-7))
                for row, b in zip(matrix, bound)
            ]
        )
    )
    rational_status = str(rational.check())

    def check(items, preserve_behavior=False):
        nonlocal calls
        correct = total = 0
        counterexamples = []
        for chunk, ids in batches(items, 128):
            predictions = edited(ids, use_cache=False).logits[:, -1].argmax(-1).tolist()
            calls += 1
            originals = (
                model(ids, use_cache=False).logits[:, -1].argmax(-1).tolist()
                if preserve_behavior
                else None
            )
            if preserve_behavior:
                calls += 1
            for j, (row, pred) in enumerate(zip(chunk, predictions)):
                expected = (
                    originals[j]
                    if preserve_behavior
                    else rule_predict(rule, row["input_ids"])
                )
                correct += pred == expected
                total += 1
                if pred != expected and len(counterexamples) < 10:
                    counterexamples.append(
                        {
                            "input_ids": row["input_ids"],
                            "actual": pred,
                            "expected": expected,
                        }
                    )
        return {
            "checked": total,
            "accuracy": correct / total if total else None,
            "counterexamples": counterexamples,
        }

    synthesis_check = check(domain)
    validation = check(verification) if verification else None
    preservation = check(preserve, True) if preserve else None
    passed = (
        synthesis_check["accuracy"] == 1.0
        and (validation is None or validation["accuracy"] == 1.0)
        and (preservation is None or preservation["accuracy"] == 1.0)
    )
    report = {
        "backend": "gpt2.rank_one_constraint_compiler",
        "rule": rule,
        "passed": passed,
        "source_sha256": fingerprint(model),
        "edited_sha256": fingerprint(edited),
        "audit": audit(model, edited),
        "synthesis": synthesis_check,
        "verification": validation,
        "preservation": preservation,
        "synthesis_domain_sha256": hashlib.sha256(
            json.dumps(domain, sort_keys=True).encode()
        ).hexdigest(),
        "coefficients": coefficients.tolist(),
        "direction": direction.tolist(),
        "nonzero_neuron_coefficients": int(
            np.count_nonzero(np.abs(coefficients[:-1]) > 1e-9)
        ),
        "solver": {
            "lp_status": result.message,
            "linear_constraint_count": len(bound),
            "rational_violation_check": rational_status,
            "claim": "rational check of recorded LP inequalities with 1e-7 slack; separate native float32 domain verification",
        },
        "forward_calls": calls,
        "gradient_updates": 0,
        "limitations": "activation-informed constrained weight synthesis; not weight-only decompilation, not zero-data editing, not universal equivalence or a new language capability",
    }
    if not passed and strict:
        raise ValueError(f"native execution rejected synthesized program: {report}")
    return edited, report


def cegis(model, rule, initial_domain, oracle_domain, heldout=None, max_rounds=4):
    """Counterexample-guided bounded synthesis; held-out evaluation happens last."""
    from .evaluate import score

    if max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    training = copy.deepcopy(initial_domain)
    history = []
    calls = 0
    if heldout and {tuple(r["input_ids"]) for r in initial_domain + oracle_domain} & {
        tuple(r["input_ids"]) for r in heldout
    }:
        raise ValueError("held-out domain overlaps synthesis/oracle domains")
    edited = None
    for iteration in range(max_rounds):
        seen = {tuple(r["input_ids"]) for r in training}
        oracle = [r for r in oracle_domain if tuple(r["input_ids"]) not in seen]
        edited, report = compile_rule(
            model, rule, training, verification=oracle, strict=False
        )
        calls += report["forward_calls"]
        history.append(report)
        if report["passed"]:
            break
        counterexamples = (
            report["verification"]["counterexamples"] if report["verification"] else []
        )
        if not counterexamples:
            break
        for example in counterexamples:
            if tuple(example["input_ids"]) not in seen:
                training.append({"input_ids": example["input_ids"]})
                seen.add(tuple(example["input_ids"]))
    result = {
        "backend": "counterexample_guided_constraint_compiler",
        "rule": rule,
        "rounds": len(history),
        "history": history,
        "bounded_oracle_passed": bool(history and history[-1]["passed"]),
        "source_sha256": fingerprint(model),
        "audit": audit(model, edited),
        "forward_calls": calls,
        "final_synthesis_examples": len(training),
        "oracle_examples": len(oracle_domain),
        "claim": "bounded activation-informed synthesis; held-out rows are never supplied to LP or counterexample refinement",
    }
    if heldout:
        labeled = [
            dict(row, target=rule_predict(rule, row["input_ids"])) for row in heldout
        ]
        result["base_heldout"] = score(model, labeled, 128)
        result["edited_heldout"] = score(edited, labeled, 128, reference=model)
    return edited, result
