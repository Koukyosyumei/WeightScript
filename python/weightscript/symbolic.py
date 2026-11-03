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
