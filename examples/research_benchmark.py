"""Run a real trained-checkpoint pilot, with raw-weight and program-edit arms.

No optimizer or gradient computation. This is a controlled editing benchmark,
not a replication of weight-space-programming's frontier-LLM experiment.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from weightscript.core import extract, load_model, write_json, compile_edits, audit
from weightscript.evaluate import benchmark, equivalence


def rows(seed, lengths, count=32):
    rng = random.Random(seed)
    result = []
    for length in lengths:
        seen = set()
        while len(seen) < count:
            # Balanced range of class margins, like upstream's binary-majority dataset.
            zero = rng.randint(0, length)
            bits = [0] * zero + [1] * (length - zero)
            rng.shuffle(bits)
            ids = tuple([2] + bits + [3])  # upstream BOS=2, SEP=3
            if ids in seen:
                continue
            seen.add(ids)
            result.append({"input_ids": list(ids), "target": int(zero < length - zero)})
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", help="upstream bin_majority trained checkpoint")
    p.add_argument("--output", default="artifacts/research")
    p.add_argument("--budget", type=int, default=48)
    args = p.parse_args()
    torch.set_num_threads(2)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model)
    dev = rows(710, [15, 31, 63])
    test = rows(928, [17, 33, 65])
    for name, data in [("dev", dev), ("heldout", test)]:
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))
    base = extract(model)
    write_json(out / "program.json", base)
    # Same proposals expressed in two interfaces: an essential parity control.
    # A method advantage cannot be inferred from these identical interventions.
    candidates = {
        "direct_tensor_control": [{"raw": []}],
        "program_control": [{"edits": []}],
    }
    for factor in [0, 0.25, 0.5, 0.75, 1.25, 1.5, 2.0]:
        for layer in range(model.config.n_layer):
            head_width = model.config.n_embd // model.config.n_head
            for head in range(model.config.n_head):
                candidates["direct_tensor_control"].append(
                    {
                        "raw": [
                            {
                                "op": "scale",
                                "tensor": f"transformer.h.{layer}.attn.c_proj.weight",
                                "rows": list(
                                    range(head * head_width, (head + 1) * head_width)
                                ),
                                "factor": factor,
                            }
                        ]
                    }
                )
                candidates["program_control"].append(
                    {
                        "edits": [
                            {
                                "op": "head_scale",
                                "layer": layer,
                                "head": head,
                                "factor": factor,
                            }
                        ]
                    }
                )
    write_json(out / "candidates.json", candidates)
    report = benchmark(model, candidates, dev, test, batch_size=32, budget=args.budget)
    report["checkpoint_source"] = {
        "repository": "https://github.com/lacoco-lab/decompiling_transformers",
        "revision": "787398b0f4393a90ad02cd8e71be48d1e85d2b21",
        "checkpoint": "bin_majority-@1l1h16d3lr01drop",
        "status": "gradient-trained upstream checkpoint; no training in this experiment",
    }
    report["experiment"] = (
        "matched-intervention parity control on binary-majority prediction, development lengths 15/31/63 and held-out lengths 17/33/65"
    )
    report["conclusion"] = (
        "These arms express identical edits; parity validates the compiler, not superiority. A learned proposal policy and independent task suite are needed to compare effectiveness."
    )
    write_json(out / "benchmark.json", report)
    selected = report["methods"][1]["program"]
    write_json(out / "selected.program.json", selected)
    check = equivalence(model, selected, test)
    write_json(out / "equivalence.json", check)
    edited = compile_edits(model, selected)
    edited.save_pretrained(out / "checkpoint")
    # Verify actual disk round trip, not just an in-memory edited module.
    reloaded = load_model(out / "checkpoint")
    write_json(out / "reload-audit.json", audit(edited, reloaded))
    print(
        json.dumps(
            {
                "base": report["base_heldout"],
                "methods": [
                    {
                        "method": m["method"],
                        "heldout": m["heldout"],
                        "audit": m["audit"],
                    }
                    for m in report["methods"]
                ],
                "equivalence": check,
            },
            indent=2,
        )
    )
    if not check["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
