"""Behavioral primitive synthesis with explicit domain-limited evidence.

This is not weight-only reverse engineering. Unknown behavior remains opaque,
while matched hypotheses can be lowered through the real Tracr backend.
"""

import copy

import torch

from .core import fingerprint
from .evaluate import batches
from .ir import Artifact, Node


def E(op, *args):
    return {"op": op, "args": list(args)}


X = {"var": "x"}
Y = {"var": "y"}


def previous_program(triggers=(0,), vocab=(0, 1, 2, 3), max_seq_len=6):
    width = len(vocab)
    if list(vocab) != list(range(width)):
        raise ValueError(
            "previous-token template requires contiguous integer vocabulary"
        )
    return {
        "format": "weightscript.rasp.v1",
        "vocab": list(vocab),
        "max_seq_len": max_seq_len,
        "causal": True,
        "nodes": [
            {"id": "tokens", "op": "tokens"},
            {"id": "indices", "op": "indices"},
            {
                "id": "previous_index",
                "op": "map",
                "inputs": ["indices"],
                "expr": E("max", 0, E("sub", X, 1)),
            },
            {
                "id": "selector",
                "op": "select",
                "inputs": ["indices", "previous_index"],
                "comparison": "eq",
            },
            {"id": "previous", "op": "aggregate", "inputs": ["selector", "tokens"]},
            {
                "id": "pair",
                "op": "sequence_map",
                "inputs": ["tokens", "previous"],
                "expr": E("add", E("mul", X, width), Y),
            },
            {
                "id": "result",
                "op": "map",
                "inputs": ["pair"],
                "expr": E(
                    "if",
                    E("in", E("floor_div", X, width), list(triggers)),
                    E("mod", X, width),
                    E("floor_div", X, width),
                ),
            },
        ],
        "output": "result",
    }


def count_program(threshold=0.5, invert=False, max_seq_len=17):
    # SelectorWidth keeps all integer counts in Tracr's inferred basis. Numeric
    # Aggregate's upstream basis inference omits general numerator/denominator
    # fractions, so lowering a mean-threshold Map that way is unsound.
    decision = E("gt", X, E("mul", threshold, E("add", X, Y)))
    if invert:
        decision = E("not", decision)
    return {
        "format": "weightscript.rasp.v1",
        "vocab": [0, 1, 2, 3],
        "max_seq_len": max_seq_len,
        "causal": True,
        "observation": "last",
        "nodes": [
            {"id": "tokens", "op": "tokens"},
            {"id": "ones", "op": "map", "inputs": ["tokens"], "expr": E("eq", X, 1)},
            {"id": "zeros", "op": "map", "inputs": ["tokens"], "expr": E("eq", X, 0)},
            {"id": "queries", "op": "map", "inputs": ["tokens"], "expr": True},
            {
                "id": "select_ones",
                "op": "select",
                "inputs": ["ones", "queries"],
                "comparison": "eq",
            },
            {
                "id": "select_zeros",
                "op": "select",
                "inputs": ["zeros", "queries"],
                "comparison": "eq",
            },
            {"id": "count_ones", "op": "selector_width", "inputs": ["select_ones"]},
            {"id": "count_zeros", "op": "selector_width", "inputs": ["select_zeros"]},
            {
                "id": "result",
                "op": "sequence_map",
                "inputs": ["count_ones", "count_zeros"],
                "expr": E("if", decision, 1, 0),
            },
        ],
        "output": "result",
    }


@torch.inference_mode()
def synthesize_binary(
    model,
    calibration,
    evaluation=None,
    batch_size=32,
    min_fidelity=0.99,
    causal_artifact=None,
):
    """Search a declared library; evaluate selected hypothesis on untouched inputs."""
    if not calibration or not 0 <= min_fidelity <= 1:
        raise ValueError("invalid synthesis domain")
    if evaluation and {tuple(r["input_ids"]) for r in calibration} & {
        tuple(r["input_ids"]) for r in evaluation
    }:
        raise ValueError("synthesis calibration and evaluation overlap")
    hypotheses = []
    for threshold in (0.25, 0.4, 0.5, 0.6, 0.75):
        for invert in (False, True):
            hypotheses.append(
                {
                    "primitive": "count_threshold",
                    "threshold": threshold,
                    "invert": invert,
                }
            )
    hypotheses += [
        {"primitive": name}
        for name in ("first_bit", "last_bit", "parity", "constant_zero", "constant_one")
    ]

    def predict(h, ids):
        bits = [x for x in ids if x in (0, 1)]
        if not bits:
            raise ValueError(
                "binary synthesis requires at least one binary token per example"
            )
        kind = h["primitive"]
        if kind == "count_threshold":
            return int((sum(bits) / len(bits) > h["threshold"]) != h["invert"])
        if kind == "first_bit":
            return bits[0]
        if kind == "last_bit":
            return bits[-1]
        if kind == "parity":
            return sum(bits) % 2
        return int(kind == "constant_one")

    def observations(rows):
        examples = []
        calls = 0
        for chunk, ids in batches(rows, batch_size):
            pred = model(ids, use_cache=False).logits[:, -1].argmax(-1).tolist()
            calls += 1
            examples.extend(zip(chunk, pred))
        return examples, calls

    observations_dev, calls = observations(calibration)
    scores = [
        sum(predict(h, row["input_ids"]) == pred for row, pred in observations_dev)
        / len(calibration)
        for h in hypotheses
    ]
    selected = copy.deepcopy(
        hypotheses[max(range(len(hypotheses)), key=lambda i: scores[i])]
    )
    fidelity = max(scores)
    test = None
    test_calls = 0
    if evaluation:
        observations_test, test_calls = observations(evaluation)
        test = sum(
            predict(selected, row["input_ids"]) == pred
            for row, pred in observations_test
        ) / len(evaluation)
    accepted = fidelity >= min_fidelity
    nodes = [Node("tokens", "tokens", interpretation="primitive")]
    if accepted:
        nodes.append(
            Node(
                "hypothesis",
                selected.pop("primitive"),
                ["tokens"],
                selected,
                interpretation="hypothesis",
            )
        )
    else:
        nodes.append(
            Node("hypothesis", "native_model", ["tokens"], interpretation="opaque")
        )
    provenance = {
        "checkpoint_sha256": fingerprint(model),
        "method": "finite-library behavioral synthesis; model forward passes used",
        "candidate_count": len(hypotheses),
        "causal_evidence": causal_artifact,
    }
    artifact = Artifact(
        "symbolic.synthesis",
        nodes,
        "hypothesis",
        provenance,
        {
            "claim": "hypothesis on supplied domains, not a general or weight-only equivalence",
            "accepted": accepted,
            "calibration_agreement": fidelity,
            "evaluation_agreement": test,
            "calibration_examples": len(calibration),
            "evaluation_examples": len(evaluation or []),
            "forward_calls": calls + test_calls,
            "candidate_agreements": scores,
        },
    )
    return artifact


def lower(artifact, max_seq_len):
    node = (
        artifact.nodes[-1] if isinstance(artifact, Artifact) else artifact["nodes"][-1]
    )
    op = node.op if isinstance(node, Node) else node["op"]
    attrs = node.attributes if isinstance(node, Node) else node["attributes"]
    if op != "count_threshold":
        raise ValueError(f"no Tracr lowering for {op}; retain opaque computation")
    return count_program(attrs["threshold"], attrs["invert"], max_seq_len)


@torch.inference_mode()
def synthesize_rules(model, candidates, calibration, evaluation=None, *,
                     min_fidelity=1.0, batch_size=32, causal_artifact=None):
    """Match explicit semantic ASTs, retaining the numerical model as fallback.

    Candidates are selected using calibration only. Evaluation failures are saved,
    never used to choose another rule. Causal artifacts are checkpoint-bound
    evidence links; behavioral agreement alone does not identify an internal op.
    """
    from .backends.rules import rule_predict, validate_rule

    if not calibration or not candidates or not 0 <= min_fidelity <= 1:
        raise ValueError("invalid semantic synthesis domain/library")
    if len(candidates) > 1024:
        raise ValueError("semantic candidate budget exceeded")
    if evaluation and {tuple(r["input_ids"]) for r in calibration} & {
        tuple(r["input_ids"]) for r in evaluation
    }:
        raise ValueError("synthesis calibration and evaluation overlap")
    candidates = copy.deepcopy(candidates)
    for rule in candidates:
        validate_rule(rule)
    source_hash = fingerprint(model)
    evidence = causal_artifact.to_dict() if isinstance(causal_artifact, Artifact) else causal_artifact
    if evidence is not None and (not isinstance(evidence, dict)
            or evidence.get("provenance", {}).get("checkpoint_sha256") != source_hash):
        raise ValueError("causal evidence belongs to a different checkpoint")
    calls = 0

    def observe(rows):
        nonlocal calls
        observations = []
        for chunk, ids in batches(rows, batch_size):
            logits = model(ids, use_cache=False).logits[:, -1]
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite model observations")
            calls += 1
            observations.extend((row["input_ids"], int(pred))
                                for row, pred in zip(chunk, logits.argmax(-1)))
        return observations

    def measure(rule, observations):
        correct = 0
        failures = []
        for ids, expected in observations:
            try:
                actual = rule_predict(rule, ids)
                error = None
            except ValueError as exc:
                actual, error = None, str(exc)
            correct += actual == expected
            if actual != expected and len(failures) < 20:
                failures.append({"input_ids": ids, "expected": expected,
                                 "actual": actual, "error": error})
        return {"agreement": correct / len(observations),
                "examples": len(observations), "counterexamples": failures}

    observations = observe(calibration)
    metrics = [measure(rule, observations) for rule in candidates]
    selected = max(range(len(candidates)), key=lambda i: metrics[i]["agreement"])
    validation = measure(candidates[selected], observe(evaluation)) if evaluation else None
    accepted = metrics[selected]["agreement"] >= min_fidelity
    nodes = [Node("tokens", "tokens", interpretation="primitive"),
             Node("native", "native_model", ["tokens"],
                  {"checkpoint_sha256": source_hash},
                  [name for name, _ in model.named_parameters()], "opaque")]
    if accepted:
        nodes.append(Node("hypothesis", "semantic_rule", ["tokens"],
                          {"rule": candidates[selected], "fallback": "native"},
                          interpretation="hypothesis"))
    if fingerprint(model) != source_hash:
        raise RuntimeError("semantic synthesis mutated source model")
    return Artifact(
        "symbolic.semantic_synthesis", nodes, "hypothesis" if accepted else "native",
        {"checkpoint_sha256": source_hash,
         "method": "finite-library behavioral synthesis with explicit semantic ASTs",
         "causal_evidence": evidence, "candidates": candidates},
        {"claim": "bounded behavioral hypothesis; internal primitive identification unproven",
         "accepted": accepted, "selected_index": selected,
         "calibration": metrics[selected], "evaluation": validation,
         "evaluation_passed": validation["agreement"] >= min_fidelity if validation else None,
         "candidate_metrics": metrics, "forward_calls": calls})


@torch.inference_mode()
def synthesize_sequence_rules(model, candidates, calibration, evaluation=None, final=None, *,
                              min_fidelity=1.0, batch_size=32, causal_artifact=None):
    """Match explicit span programs to teacher-forced model decisions.

    Calibration chooses the candidate. Evaluation and final rows are scored
    afterwards and cannot replace the chosen rule. Agreement with the model is
    a behavioral hypothesis. Positions the library cannot explain stay on the
    opaque native model.
    """
    from .backends.rules import rule_predict, supervised_pairs, validate_rule

    if not calibration or not candidates or not 0 <= min_fidelity <= 1:
        raise ValueError("invalid sequence synthesis domain/library")
    if len(candidates) > 64:
        raise ValueError("sequence candidate budget exceeded")
    domains = {"calibration": calibration}
    if evaluation:
        domains["evaluation"] = evaluation
    if final:
        domains["final"] = final
    seen = set()
    for name, rows in domains.items():
        current = set()
        for row in rows:
            ids = row.get("input_ids")
            if not isinstance(ids, list) or not ids:
                raise ValueError(f"invalid {name} sequence")
            key = tuple(ids)
            if key in seen or key in current:
                raise ValueError(f"duplicate or overlapping {name} sequence domain")
            current.add(key)
        seen.update(current)
    candidates = copy.deepcopy(candidates)
    for rule in candidates:
        validate_rule(rule)
        if rule["op"] != "span_next":
            raise ValueError("sequence synthesis accepts span_next rules only")
    if len({(rule["bos"], rule["sep"], rule["eos"]) for rule in candidates}) != 1:
        raise ValueError("sequence candidates must share bos, sep and eos")
    sep = candidates[0]["sep"]
    source_hash = fingerprint(model)
    evidence = causal_artifact.to_dict() if isinstance(causal_artifact, Artifact) else causal_artifact
    if evidence is not None and (not isinstance(evidence, dict)
            or evidence.get("provenance", {}).get("checkpoint_sha256") != source_hash):
        raise ValueError("causal evidence belongs to a different checkpoint")
    calls = 0

    def observe(rows):
        nonlocal calls
        observed = []
        for chunk, ids in batches(rows, batch_size):
            logits = model(ids, use_cache=False).logits
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite model observations")
            calls += 1
            for row, row_logits in zip(chunk, logits):
                positions = []
                for prefix, gold in supervised_pairs(row["input_ids"], sep):
                    positions.append((prefix, gold, int(row_logits[len(prefix) - 1].argmax())))
                observed.append(positions)
        return observed

    def measure(rule, observed):
        correct = rule_task = model_task = total = 0
        failures = []
        for positions in observed:
            for prefix, gold, predicted in positions:
                try:
                    actual = rule_predict(rule, prefix)
                    error = None
                except ValueError as exc:
                    actual, error = None, str(exc)
                correct += actual == predicted
                rule_task += actual == gold
                model_task += predicted == gold
                total += 1
                if actual != predicted and len(failures) < 12:
                    failures.append({"input_ids": prefix, "model": predicted, "rule": actual,
                                     "gold": gold, "error": error})
        return {"agreement": correct / total, "rule_task_agreement": rule_task / total,
                "model_task_agreement": model_task / total, "positions": total,
                "sequences": len(observed), "counterexamples": failures}

    calibration_obs = observe(calibration)
    metrics = [measure(rule, calibration_obs) for rule in candidates]
    selected = max(range(len(candidates)), key=lambda i: (metrics[i]["agreement"], -i))
    evaluation_metric = measure(candidates[selected], observe(evaluation)) if evaluation else None
    final_metric = measure(candidates[selected], observe(final)) if final else None
    accepted = metrics[selected]["agreement"] >= min_fidelity
    nodes = [Node("tokens", "tokens", interpretation="primitive"),
             Node("native", "native_model", ["tokens"],
                  {"checkpoint_sha256": source_hash},
                  [name for name, _ in model.named_parameters()], "opaque")]
    if accepted:
        nodes.append(Node("hypothesis", "span_next", ["tokens"],
                          {"rule": candidates[selected], "fallback": "native"},
                          interpretation="hypothesis"))
    if fingerprint(model) != source_hash:
        raise RuntimeError("sequence synthesis mutated source model")
    return Artifact(
        "symbolic.sequence_synthesis", nodes, "hypothesis" if accepted else "native",
        {"checkpoint_sha256": source_hash,
         "method": "finite-library behavioral sequence synthesis; model forward passes used",
         "causal_evidence": evidence, "candidates": candidates,
         "selection": "calibration agreement only; evaluation and final are reports"},
        {"claim": "bounded behavioral sequence hypothesis; not weight-only discovery or proof of an internal algorithm",
         "accepted": accepted, "selected_index": selected,
         "calibration": metrics[selected], "evaluation": evaluation_metric, "final": final_metric,
         "evaluation_passed": evaluation_metric["agreement"] >= min_fidelity if evaluation_metric else None,
         "final_passed": final_metric["agreement"] >= min_fidelity if final_metric else None,
         "candidate_metrics": metrics, "forward_calls": calls})
