"""Run an existing Codex TUI as editing agent against a small local pretrained LM."""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer
from weightscript.agent import run_session
from weightscript.core import load_model

p = argparse.ArgumentParser()
p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
p.add_argument("--workspace", required=True)
p.add_argument("--session", required=True)
p.add_argument("--mode", choices=["direct", "program", "hybrid"], default="hybrid")
p.add_argument("--output", default="artifacts/codex-tui-study")
p.add_argument("--steps", type=int, default=8)
a = p.parse_args()
torch.set_num_threads(2)
tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
# Independently authored exploratory sentence-completion probe sets. They are
# not public benchmark scores or held-out pretraining data.
dev_pairs = [
    ("The capital of France is", " Paris"),
    ("Water freezes at zero degrees", " Celsius"),
    ("The opposite of hot is", " cold"),
    ("There are seven days in a", " week"),
    ("A triangle has three", " sides"),
    ("The largest planet is", " Jupiter"),
    ("The sun rises in the", " east"),
    ("A cat is an", " animal"),
    ("Two plus two equals", " four"),
    ("The color of a clear sky is", " blue"),
    ("Birds have", " wings"),
    ("Fish live in", " water"),
    ("The capital of Japan is", " Tokyo"),
    ("A year has twelve", " months"),
    ("Ice is frozen", " water"),
    ("The Earth orbits the", " Sun"),
]
test_pairs = [
    ("The capital of Germany is", " Berlin"),
    ("There are sixty seconds in a", " minute"),
    ("The opposite of big is", " small"),
    ("A square has four", " sides"),
    ("The closest star to Earth is the", " Sun"),
    ("The capital of Italy is", " Rome"),
    ("One plus one equals", " two"),
    ("A dog is an", " animal"),
    ("Humans breathe", " air"),
    ("Plants need", " sunlight"),
    ("The capital of Spain is", " Madrid"),
    ("The opposite of day is", " night"),
    ("A week has seven", " days"),
    ("Snow is usually", " white"),
    ("The ocean contains salt", " water"),
    ("A bicycle has two", " wheels"),
]
collateral_pairs = [
    ("Once upon a", " time"),
    ("Thank you very", " much"),
    ("How are", " you"),
    ("In the middle of the", " night"),
    ("To be or not to", " be"),
    ("I would like to", " know"),
    ("As a matter of", " fact"),
    ("At the end of the", " day"),
]


def rows(pairs):
    return [
        {
            "input_ids": tokenizer.encode(prefix),
            "target": tokenizer.encode(answer, add_special_tokens=False)[0],
        }
        for prefix, answer in pairs
    ]


policy = [
    sys.executable,
    "-m",
    "weightscript.tmux_policy",
    "--workspace",
    str(Path(a.workspace).resolve()),
    "--session",
    a.session,
    "--timeout",
    "240",
]
model = load_model(a.model)
report = run_session(
    model,
    policy,
    "Improve next-token accuracy and probability on the provided sentence-completion development set by targeted weight edits, with no gradient descent. The target is a genuine pretrained SmolLM2-135M language model. Inspect tensors or traces, form a circuit hypothesis, evaluate a few sparse candidates, and retain general language behavior. Do not access evaluation data outside the tool protocol. This is an exploratory pilot, not evidence of general benchmark superiority.",
    rows(dev_pairs),
    rows(test_pairs),
    mode=a.mode,
    max_forwards=150,
    max_steps=a.steps,
    timeout=260,
    seed=42,
    output=a.output,
    collateral=rows(collateral_pairs),
)
print(
    {
        k: report.get(k)
        for k in (
            "baseline_dev",
            "selected_dev",
            "baseline_heldout",
            "selected_heldout",
            "selected_collateral",
            "budget",
            "audit",
        )
    }
)
