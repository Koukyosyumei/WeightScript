"""Local three-stage upstream extraction and bounded semantic editing orchestration."""

import hashlib
import json

from ..core import fingerprint
from ..symbolic import synthesize_rules
from ..synthesis import cegis
from .rules import validate_rule
from .upstream_stages import decompile_stages


def disjoint_domains(**domains):
    """Validate named nonempty token domains and return reproducible hashes."""
    seen = set()
    hashes = {}
    for name, rows in domains.items():
        if not rows:
            raise ValueError(f"{name} domain is empty")
        current = set()
        for row in rows:
            ids = row.get("input_ids")
            if (
                not isinstance(ids, list)
                or not ids
                or any(type(t) is not int or t < 0 for t in ids)
            ):
                raise ValueError(f"invalid {name} input_ids")
            key = tuple(ids)
            if key in seen or key in current:
                raise ValueError(f"duplicate or overlapping {name} domain")
            current.add(key)
        seen.update(current)
        hashes[name] = hashlib.sha256(json.dumps(sorted(current)).encode()).hexdigest()
    return hashes


def summarize_stages(artifact):
    """Return measured stage and primitive counts for a three-stage artifact.

    The full artifact remains the replay source. This summary does not add
    evidence and does not treat a matrix fit as a replaced primitive.
    """
    data = artifact.to_dict() if hasattr(artifact, "to_dict") else artifact
    if (
        not isinstance(data, dict)
        or data.get("backend") != "decompiling_transformers.three_stage"
    ):
        raise ValueError("expected a three-stage artifact")
    output = data["nodes"][-1]["attributes"]
    stages = []
    for stage in output["stages"]:
        if stage["name"] == "component":
            fidelity = stage["artifact"]["fidelity"]
            evaluation = fidelity.get("evaluation")
            stages.append(
                {
                    "name": "component",
                    "accepted": stage["accepted"],
                    "edges_before": fidelity["component_edges_before"],
                    "edges_after": fidelity["component_edges_after"],
                    "trials": len(fidelity["trials"]),
                    "search_complete": len(fidelity["trials"])
                    >= fidelity["component_edges_before"],
                    "calibration_agreement": fidelity["calibration"][
                        "final_token_agreement"
                    ],
                    "evaluation_agreement": None
                    if evaluation is None
                    else evaluation["final_token_agreement"],
                    "max_logit_error": fidelity["calibration"]["max_logit_error"],
                    "logit_reconstruction": fidelity.get("logit_reconstruction"),
                    "normalization_assumption_passed": fidelity[
                        "normalization_assumption_passed"
                    ],
                }
            )
            continue
        evaluation = stage.get("evaluation")
        stages.append(
            {
                "name": stage["name"],
                "accepted": stage["accepted"],
                "edges_before": stage["edges_before"],
                "edges_after": stage["edges_after"],
                "trials": len(stage["trials"]),
                "search_complete": len(stage["trials"]) >= stage["edges_before"],
                "calibration_agreement": stage["calibration"]["final_token_agreement"],
                "evaluation_agreement": None
                if evaluation is None
                else evaluation["final_token_agreement"],
                "max_logit_error": stage["calibration"]["max_logit_error"],
                "renderer_adaptation": stage.get("renderer_adaptation"),
            }
        )
    primitives = output.get("primitive_matches", [])
    counts = {}
    hypotheses = []
    rejections = []
    for node in primitives:
        counts[node["interpretation"]] = counts.get(node["interpretation"], 0) + 1
        best = node["attributes"].get("best_match") or {}
        row = {
            "id": node["id"],
            "primitive": best.get("primitive"),
            "operation": best.get("operation"),
            "relative_error": best.get("relative_error"),
            "claim": node["attributes"].get("claim")
            or node["attributes"].get("evidence"),
        }
        if node["interpretation"] == "hypothesis":
            hypotheses.append(row)
        elif best:
            rejections.append(row)
    return {
        "accepted": data["fidelity"]["accepted"],
        "claim": data["fidelity"]["claim"],
        "checkpoint_sha256": data["provenance"]["checkpoint_sha256"],
        "completed_stages": data["provenance"]["completed_stages"],
        "compatibility_adaptations": data["provenance"].get(
            "compatibility_adaptations", []
        ),
        "normalization": data["fidelity"].get("normalization"),
        "stages": stages,
        "primitive_interpretations": counts,
        "primitive_hypotheses": hypotheses,
        "primitive_rejections": rejections,
    }


def decompile_and_edit(
    model,
    checkout,
    candidates,
    target_rule,
    synthesis_rows,
    oracle_rows,
    final_rows,
    *,
    max_trials=64,
    max_rounds=4,
    min_fidelity=1.0,
    calibration_steps=32,
    require_causal_fidelity=True,
):
    """Return edited model and JSON evidence; final rows are scored only last.

    The proposed target rule is explicit. Matching the old behavior does not
    authorize a claim that its AST describes the learned internal algorithm.
    """
    hashes = disjoint_domains(
        synthesis=synthesis_rows, oracle=oracle_rows, final=final_rows
    )
    validate_rule(target_rule)
    source_hash = fingerprint(model)
    causal = decompile_stages(
        model,
        checkout,
        synthesis_rows,
        max_trials=max_trials,
        min_fidelity=min_fidelity,
        evaluation_rows=oracle_rows,
        calibration_steps=calibration_steps,
    )
    if require_causal_fidelity and not causal.fidelity["accepted"]:
        return None, {
            "status": "causal_rejected",
            "source_sha256": source_hash,
            "domains": hashes,
            "causal": causal.to_dict(),
        }
    hypothesis = synthesize_rules(
        model,
        candidates,
        synthesis_rows,
        oracle_rows,
        min_fidelity=min_fidelity,
        causal_artifact=causal,
    )
    if (
        not hypothesis.fidelity["accepted"]
        or not hypothesis.fidelity["evaluation_passed"]
    ):
        return None, {
            "status": "hypothesis_rejected",
            "source_sha256": source_hash,
            "domains": hashes,
            "causal": causal.to_dict(),
            "hypothesis": hypothesis.to_dict(),
        }
    edited, result = cegis(
        model,
        target_rule,
        synthesis_rows,
        oracle_rows,
        heldout=final_rows,
        max_rounds=max_rounds,
    )
    if fingerprint(model) != source_hash:
        raise RuntimeError("semantic pipeline mutated source weights")
    return edited, {
        "status": "evaluated",
        "source_sha256": source_hash,
        "domains": hashes,
        "causal": causal.to_dict(),
        "hypothesis": hypothesis.to_dict(),
        "edit": result,
        "completed_upstream_stages": ["component", "full_path", "qk"],
        "causal_fidelity_accepted": causal.fidelity["accepted"],
    }
