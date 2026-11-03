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

from .backends.rules import rule_predict, validate_rule
from .core import audit, fingerprint
from .evaluate import batches


@torch.no_grad()
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
    max_variables=100000,
    max_constraint_entries=5000000,
):
    validate_rule(rule)
    import z3
    from scipy.optimize import linprog

    if (
        model.config.model_type != "gpt2"
        or not domain
        or margin <= 0
        or coefficient_bound <= 0
        or not np.isfinite([margin, coefficient_bound, timeout]).all()
        or timeout <= 0
        or type(max_variables) is not int or max_variables < 1
        or type(max_constraint_entries) is not int or max_constraint_entries < 1
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
    if verification and preserve and {tuple(r["input_ids"]) for r in verification} & {
        tuple(r["input_ids"]) for r in preserve
    }:
        raise ValueError("verification and preservation domains overlap")
    original_hash = fingerprint(model)
    if model.training or next(model.parameters()).device.type != "cpu":
        raise ValueError("constraint compiler requires an eval-mode CPU model")
    rows = domain + (preserve or [])
    snapshots = []
    calls = 0
    for chunk, ids in batches(rows, 128):
        hidden = []
        residual = []
        h = model.transformer.h[-1].mlp.c_proj.register_forward_pre_hook(
            lambda _, args, hidden=hidden: hidden.append(args[0][:, -1].detach().clone())
        )
        r = model.transformer.ln_f.register_forward_pre_hook(
            lambda _, args, residual=residual: residual.append(args[0][:, -1].detach().clone())
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
    targets = sorted({target for _, _, target in snapshots})
    if any(target < 0 or target >= model.config.vocab_size for target in targets):
        raise ValueError("rule output outside model vocabulary")
    # Keep the historical binary rank-one patch; other rules use the span of
    # target unembedding contrasts. This supports multiclass copy/selection.
    if rule["op"] == "count_threshold":
        contrasts = ((unembed[1] - unembed[0]) * gamma)[None, :]
    else:
        anchor = next((i for i in range(model.config.vocab_size) if i not in targets), 0)
        contrasts = torch.stack([(unembed[t] - unembed[anchor]) * gamma for t in targets])
    contrasts = contrasts - contrasts.mean(-1, keepdim=True)
    _, singular, vh = torch.linalg.svd(contrasts.double(), full_matrices=False)
    rank = int((singular > max(float(singular.max()) * 1e-7, 1e-12)).sum())
    if not rank:
        raise ValueError("output contrasts have no residual direction")
    directions = vh[:rank].float()
    feature_count = snapshots[0][0].numel() + 1
    variable_count = feature_count * rank
    projected_entries = variable_count * len(snapshots) * (model.config.vocab_size - 1)
    if variable_count > max_variables or projected_entries > max_constraint_entries:
        raise ValueError(
            "constraint compiler variable budget exceeded: "
            f"variables={variable_count}, snapshots={len(snapshots)}, "
            f"projected_entries={projected_entries}, rank={rank}")
    inequalities = []
    upper = []
    for hidden, residual, target in snapshots:
        features = np.append(hidden.double().numpy(), 1.0)
        for other in range(model.config.vocab_size):
            if other == target:
                continue
            diff = ((unembed[target] - unembed[other]) * gamma).double()
            diff -= diff.mean()
            inequalities.append(-np.outer((directions.double() @ diff).numpy(), features).reshape(-1))
            upper.append(float(residual.double() @ diff) - margin)
    matrix = np.array(inequalities)
    bound = np.array(upper)
    # Positive/negative split minimizes coefficient L1 and often uses few units.
    result = linprog(
        np.ones(2 * variable_count),
        A_ub=np.concatenate((matrix, -matrix), axis=1),
        b_ub=bound,
        bounds=[(0, coefficient_bound)] * (2 * variable_count),
        method="highs",
        options={"time_limit": timeout},
    )
    if not result.success:
        raise ValueError(f"rule does not fit chosen output-span patch: {result.message}")
    coefficients = result.x[:variable_count] - result.x[variable_count:]
    edited = copy.deepcopy(model).eval()
    co = torch.tensor(coefficients.reshape(rank, feature_count), dtype=torch.float32)
    edited.transformer.h[-1].mlp.c_proj.weight.add_(co[:, :-1].T @ directions)
    edited.transformer.h[-1].mlp.c_proj.bias.add_(co[:, -1] @ directions)
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
                        for a, c in zip(row, coefficients) if a != 0 and c != 0
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
        "backend": "gpt2.rank_one_constraint_compiler" if rank == 1 else "gpt2.multidirection_constraint_compiler",
        "rule": rule,
        "passed": passed,
        "source_sha256": original_hash,
        "edited_sha256": fingerprint(edited),
        "audit": audit(model, edited),
        "synthesis": synthesis_check,
        "verification": validation,
        "preservation": preservation,
        "synthesis_domain_sha256": hashlib.sha256(
            json.dumps(domain, sort_keys=True).encode()
        ).hexdigest(),
        "coefficients": coefficients.tolist(),
        "direction": directions[0].tolist() if rank == 1 else None,
        "directions": directions.tolist(),
        "patch_rank_bound": rank,
        "coefficient_shape": [rank, feature_count],
        "nonzero_neuron_coefficients": int(
            np.count_nonzero(np.abs(co[:, :-1].numpy()) > 1e-9)
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
    if fingerprint(model) != original_hash:
        raise RuntimeError("constraint compilation mutated source model")
    if not passed and strict:
        raise ValueError(f"native execution rejected synthesized program: {report}")
    return edited, report


@torch.no_grad()
def _decision_check(model, rule, rows, limit=8):
    """Native last-token agreement, with a few stored mismatches."""
    if not rows:
        return {"checked": 0, "accuracy": None, "counterexamples": []}
    correct = 0
    failures = []
    for chunk, ids in batches(rows, 128):
        predictions = model(ids, use_cache=False).logits[:, -1].argmax(-1).tolist()
        for row, pred in zip(chunk, predictions):
            expected = rule_predict(rule, row["input_ids"])
            correct += pred == expected
            if pred != expected and len(failures) < limit:
                failures.append({"input_ids": row["input_ids"], "actual": pred, "expected": expected})
    return {"checked": len(rows), "accuracy": correct / len(rows), "counterexamples": failures}


def cegis(model, rule, initial_domain, oracle_domain, heldout=None, max_rounds=4):
    """Counterexample-guided bounded synthesis; held-out evaluation happens last."""
    from .evaluate import score

    validate_rule(rule)
    if not initial_domain or not oracle_domain:
        raise ValueError("synthesis and oracle domains must be nonempty")
    if {tuple(r["input_ids"]) for r in initial_domain} & {tuple(r["input_ids"]) for r in oracle_domain}:
        raise ValueError("initial synthesis and oracle domains overlap")
    if type(max_rounds) is not int or max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    training = copy.deepcopy(initial_domain)
    history = []
    calls = 0
    if heldout and {tuple(r["input_ids"]) for r in initial_domain + oracle_domain} & {
        tuple(r["input_ids"]) for r in heldout
    }:
        raise ValueError("held-out domain overlaps synthesis/oracle domains")
    edited = None
    stopped = None
    for _iteration in range(max_rounds):
        seen = {tuple(r["input_ids"]) for r in training}
        oracle = [r for r in oracle_domain if tuple(r["input_ids"]) not in seen]
        try:
            proposal, report = compile_rule(
                model, rule, training, verification=oracle or None, strict=False
            )
        except ValueError as exc:
            # A later round can exceed the LP budget. Keep the last solved patch.
            stopped = str(exc)
            break
        edited = proposal
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
    oracle_rescore = _decision_check(edited, rule, oracle_domain) if edited is not None else None
    if oracle_rescore is not None:
        calls += max(1, (len(oracle_domain) + 127) // 128)
    result = {
        "backend": "counterexample_guided_constraint_compiler",
        "rule": rule,
        "rounds": len(history),
        "history": history,
        "bounded_oracle_passed": bool(oracle_rescore and oracle_rescore["accuracy"] == 1.0),
        "oracle_rescore": oracle_rescore,
        "oracle_rows_absorbed": len(training) - len(initial_domain),
        "refinement_stopped": stopped,
        "source_sha256": fingerprint(model),
        "audit": None if edited is None else audit(model, edited),
        "forward_calls": calls,
        "final_synthesis_examples": len(training),
        "oracle_examples": len(oracle_domain),
        "claim": "bounded activation-informed synthesis; held-out rows are never supplied to LP or counterexample refinement",
    }
    if heldout and edited is not None:
        labeled = [
            dict(row, target=rule_predict(rule, row["input_ids"])) for row in heldout
        ]
        result["base_heldout"] = score(model, labeled, 128)
        result["edited_heldout"] = score(edited, labeled, 128, reference=model)
        result["heldout_decisions"] = _decision_check(edited, rule, heldout)
        result["forward_calls"] += (result["base_heldout"]["forward_calls"]
                                    + result["edited_heldout"]["forward_calls"]
                                    + result["edited_heldout"]["reference_forward_calls"]
                                    + max(1, (len(heldout) + 127) // 128))
    return edited, result


@torch.no_grad()
def replay_patch(model, report):
    """Apply a serialized compiler result without probes, solvers or training.

    The source and resulting hashes bind all weights; shape/finite checks happen
    before mutation of a private copy. The original model is never changed.
    """
    if (report.get("backend") not in ("gpt2.rank_one_constraint_compiler",
                                       "gpt2.multidirection_constraint_compiler")
            or report.get("source_sha256") != fingerprint(model)):
        raise ValueError("patch report belongs to a different checkpoint/backend")
    rank = report.get("patch_rank_bound")
    feature_count = model.transformer.h[-1].mlp.c_proj.weight.shape[0] + 1
    if (type(rank) is not int or not 1 <= rank <= model.config.n_embd
            or report.get("coefficient_shape") != [rank, feature_count]):
        raise ValueError("invalid serialized patch shape")
    directions = torch.tensor(report["directions"], dtype=torch.float32)
    coefficients = torch.tensor(report["coefficients"], dtype=torch.float32)
    if (directions.shape != (rank, model.config.n_embd)
            or coefficients.shape != (rank * feature_count,)
            or not torch.isfinite(directions).all() or not torch.isfinite(coefficients).all()):
        raise ValueError("invalid serialized patch values")
    coefficients = coefficients.reshape(rank, feature_count)
    edited = copy.deepcopy(model).eval()
    edited.transformer.h[-1].mlp.c_proj.weight.add_(coefficients[:, :-1].T @ directions)
    edited.transformer.h[-1].mlp.c_proj.bias.add_(coefficients[:, -1] @ directions)
    edited.transformer.ln_f.bias.zero_()
    if fingerprint(edited) != report.get("edited_sha256"):
        raise ValueError("serialized patch result checksum mismatch")
    return edited
