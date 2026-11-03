"""Measured local-agent comparison, with no presumed winning method."""

import argparse
import itertools
import random
import sys
from pathlib import Path

import torch
from weightscript.agent import run_session
from weightscript.core import load_model, write_json

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("--output", default="artifacts/local-agent-study")
p.add_argument("--policy-model", default="Qwen/Qwen2.5-0.5B-Instruct")
p.add_argument("--steps", type=int, default=2)
a = p.parse_args()
torch.set_num_threads(2)
root = Path(a.output)
root.mkdir(parents=True, exist_ok=True)
model = load_model(a.model)
rng = random.Random(37)
all_dev = list(itertools.product((0, 1), repeat=7))
rng.shuffle(all_dev)


def rows(bits):
    return [
        {"input_ids": [2] + list(b) + [3], "target": int(sum(b) < len(b) / 2)}
        for b in bits
    ]


dev = rows(all_dev[:16])
heldout = rows(list(itertools.product((0, 1), repeat=9)))
policy = [
    sys.executable,
    str(Path(__file__).with_name("local_policy.py")),
    "--model",
    a.policy_model,
    "--max-new-tokens",
    "256",
]
results = []
for mode in ("direct", "program"):
    report = run_session(
        model,
        policy,
        "Change this trained binary-majority predictor to output the MINORITY bit. Inputs are BOS=2, bits 0/1, SEP=3; output is the next token 0 or 1. Preserve architecture and use no gradients.",
        dev,
        heldout,
        mode=mode,
        max_forwards=12,
        max_steps=a.steps,
        timeout=180,
        seed=37,
        output=root / mode,
    )
    results.append(report)
write_json(
    root / "comparison.json",
    {
        "protocol": "same local policy model, seed, development data, step limit and tool-inference budget",
        "results": results,
        "claim": "small local instruction-agent case study; not a reproduction or defeat of frontier-agent benchmark",
    },
)
for report in results:
    print(report["mode"], report["selected_heldout"], report["budget"])
