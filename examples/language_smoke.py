"""Exercise extraction, program edits, checkpoint export/reload and undo on a pretrained LM.

The edit is an instrumentation smoke test, not a claimed capability improvement.
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer
from weightscript.core import (
    extract,
    compile_edits,
    load_model,
    audit,
    fingerprint,
    write_json,
)
from weightscript.evaluate import equivalence
from weightscript.checkpoint import save_undo, undo

PROMPTS = [
    "The capital of France is",
    "A computer program can",
    "Once upon a time,",
    "To add two numbers,",
    "The opposite of hot is",
    "Neural networks use",
    "In the morning I",
    "The quick brown fox jumps over the",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?", default="distilbert/distilgpt2")
    p.add_argument("--output", default="artifacts/language-smoke")
    args = p.parse_args()
    torch.set_num_threads(2)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "checkpoint"
    if checkpoint.exists():
        raise ValueError("choose a fresh output directory")
    model = load_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    rows = [{"input_ids": tokenizer.encode(prompt)} for prompt in PROMPTS]
    program = extract(model)
    # Both a head edit and a sparse selected-neuron projection are compiled.
    layer = (
        model.config.n_layer
        if model.config.model_type == "gpt2"
        else model.config.num_hidden_layers
    ) // 2
    direction = (
        (
            model.transformer.h[layer].mlp.c_proj.weight[0]
            if model.config.model_type == "gpt2"
            else model.model.layers[layer].mlp.down_proj.weight[:, 0]
        )
        .detach()
        .tolist()
    )
    program["edits"] = [
        {"op": "head_scale", "layer": 0, "head": 0, "factor": 0.5},
        {
            "op": "remove_direction",
            "layer": layer,
            "neurons": [0, 1, 2, 3],
            "direction": direction,
            "strength": 0.5,
        },
    ]
    write_json(out / "program.json", program)
    (out / "inputs.jsonl").write_text(
        "".join(__import__("json").dumps(r) + "\n" for r in rows)
    )
    report = equivalence(model, program, rows)
    edited = compile_edits(model, program)
    edited.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    save_undo(model, edited, checkpoint)
    del edited
    reloaded = load_model(checkpoint)
    restored = undo(reloaded, checkpoint)
    report.update(
        {
            "model": args.model,
            "upstream_revision": getattr(model.config, "_commit_hash", None),
            "parameters": sum(p.numel() for p in model.parameters()),
            "undo_restored_original_hash": fingerprint(restored) == fingerprint(model),
            "reload_audit": audit(compile_edits(model, program), reloaded),
            "purpose": "real pretrained-language-model integration smoke test; no improvement claim",
        }
    )
    write_json(out / "report.json", report)
    print(__import__("json").dumps(report, indent=2))
    if (
        not report["passed"]
        or not report["undo_restored_original_hash"]
        or report["reload_audit"]["changed_scalars"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
