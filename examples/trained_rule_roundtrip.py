"""Learned weights -> upstream pruned circuit -> matched rule -> edited weights.

Two lowering paths: Tracr creates a new model; constraint compilation patches
an existing trained GPT-2. Every approximation and bounded domain is recorded.
"""

import argparse
import itertools
import json
from pathlib import Path

import torch
from weightscript.backends.upstream import decompile
from weightscript.backends.tracr import compile_program
from weightscript.checkpoint import save_undo, undo
from weightscript.core import load_model, write_json, fingerprint, audit
from weightscript.symbolic import synthesize_binary, lower
from weightscript.synthesis import cegis


def domain(lengths):
    return [
        {"input_ids": [2] + list(bits) + [3]}
        for n in lengths
        for bits in itertools.product((0, 1), repeat=n)
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--checkout", required=True)
    p.add_argument("--output", default="artifacts/trained-roundtrip")
    args = p.parse_args()
    torch.set_num_threads(2)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("choose a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model)
    calibration = domain((1, 3, 5, 7))
    evaluation = domain((9,))
    causal = decompile(
        model, args.checkout, calibration, max_trials=64, evaluation_rows=evaluation
    )
    write_json(out / "upstream.json", causal.to_dict())
    hypothesis = synthesize_binary(
        model, calibration, evaluation, causal_artifact=causal.to_dict()
    )
    write_json(out / "hypothesis.json", hypothesis.to_dict())
    if (
        hypothesis.nodes[-1].op != "count_threshold"
        or not hypothesis.fidelity["accepted"]
    ):
        raise ValueError("count rule was not recovered from this checkpoint")
    original = lower(hypothesis, 13)
    edited_artifact = hypothesis.to_dict()
    edited_artifact["nodes"][-1]["attributes"]["invert"] = True
    changed = lower(edited_artifact, 13)
    write_json(out / "before.rasp.json", original)
    write_json(out / "after.rasp.json", changed)
    # New-architecture path through the actual upstream compiler.
    tracr_before, _ = compile_program(original)
    tracr_after, _ = compile_program(changed)
    if not (out / "tracr-before").exists():
        tracr_before.save(out / "tracr-before")
    if not (out / "tracr-after").exists():
        tracr_after.save(out / "tracr-after")
    compiled_check = []
    for row in evaluation:
        seq = row["input_ids"]
        expected = int(sum(t == 1 for t in seq) < sum(t == 0 for t in seq))
        compiled_check.append(tracr_after.predict(seq)[-1] == expected)
    # Existing-architecture path: explicitly bounded constraint compilation.
    rule = {"op": "count_threshold", "threshold": 0.5, "invert": True}
    edited, report = cegis(model, rule, calibration, evaluation, heldout=domain((11,)))
    edited.save_pretrained(out / "patched")
    save_undo(model, edited, out / "patched")
    loaded = load_model(out / "patched")
    restored = undo(loaded, out / "patched")
    report["upstream_causal_fidelity"] = causal.fidelity
    report["symbolic_fidelity"] = hypothesis.fidelity
    report["tracr_new_model"] = {
        "checked": len(compiled_check),
        "accuracy": sum(compiled_check) / len(compiled_check),
        "architecture_preserved": False,
    }
    report["reload_changes"] = audit(edited, loaded)["changed_scalars"]
    report["undo_restored_original"] = fingerprint(restored) == fingerprint(model)
    write_json(out / "report.json", report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("coefficients", "direction", "upstream_causal_fidelity")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
