"""Public-data editing comparison for the pinned upstream tasks and one small LM.

Development rows select each arm. Held-out and collateral rows are scored once.
Direct and program arms use the same head-scale grid. This script does not declare
a winning method from a small accuracy or NLL difference.
"""

import argparse
import gc
import json
import os
import subprocess
from pathlib import Path

import torch

from weightscript.comparison import TASKS, compare, task_rows, token_windows
from weightscript.core import fingerprint, load_model

PASSAGE = Path(__file__).with_name("data") / "public_domain_lm.json"


def revision(checkout):
    rev = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    expected = TASKS["bin_majority"]["revision"]
    if rev != expected:
        raise SystemExit(f"expected upstream revision {expected}, got {rev}")
    return rev


def splits(task, seed):
    if task == "bin_majority":
        return (
            task_rows(task, seed, [5, 7], 8),
            task_rows(task, seed + 100, [11, 13], 8),
            task_rows(task, seed + 200, [4], 6),
        )
    if task in {"unique_copy", "sort"}:
        return (
            task_rows(task, seed, [4, 5], 6),
            task_rows(task, seed + 100, [8], 8),
            task_rows(task, seed + 200, [3], 6),
        )
    raise SystemExit(f"unsupported task {task}")


def compact(report):
    return [
        {
            "method": row["method"],
            "selected_candidate": row["selected_candidate"],
            "candidates_evaluated": row["candidates_evaluated"],
            "changed_scalars": row["changed_scalars"],
            "development_accuracy": row["development"]["accuracy"],
            "development_nll": row["development"]["target_nll"],
            "heldout_accuracy": row["heldout"]["accuracy"],
            "heldout_nll": row["heldout"]["target_nll"],
            "heldout_wilson_95": row["heldout"]["accuracy_wilson_95"],
            "collateral_disagreement": row["collateral"]["token_disagreement"],
            "collateral_kl": row["collateral"]["reference_kl_per_token"],
            "development_forward_calls": row["development_forward_calls"],
            "heldout_forward_calls": row["heldout"]["forward_calls"],
            "seconds": row["seconds"],
        }
        for row in report["methods"]
    ]


def consistent(seed_rows):
    """True only when a method beats untouched on every seed, on both metrics."""
    names = [row["method"] for row in seed_rows[0]]
    winners = []
    for name in names:
        if name == "untouched":
            continue
        better = True
        for rows in seed_rows:
            arm = next(row for row in rows if row["method"] == name)
            base = next(row for row in rows if row["method"] == "untouched")
            if not (
                arm["heldout_accuracy"] > base["heldout_accuracy"]
                and arm["heldout_nll"] < base["heldout_nll"]
            ):
                better = False
                break
        if better:
            winners.append(name)
    return winners


def run_task(model, task, seeds, trials):
    rows = []
    for seed in range(seeds):
        dev, heldout, collateral = splits(task, seed)
        report = compare(model, dev, heldout, collateral, seed=seed, trials=trials)
        rows.append({"seed": seed, "methods": compact(report)})
    return {
        "provenance": {
            key: TASKS[task][key]
            for key in ("repository", "revision", "source", "checkpoint")
        },
        "seeds": rows,
        "methods_better_than_untouched_on_every_seed": consistent(
            [row["methods"] for row in rows]
        ),
        "interpretation": (
            "Held-out accuracy and NLL must both improve on every seed before a "
            "method is listed. A single-seed or NLL-only change is not superiority."
        ),
    }


def run_language(checkpoint, seeds, trials):
    from transformers import AutoTokenizer

    from weightscript.checkpoint import undo

    path = Path(checkpoint)
    edited = load_model(path)
    if (path / "undo.json").is_file():
        model = undo(edited, path)
        restoration = (
            "undo package restored the checkpoint that existed before the saved edit"
        )
    else:
        model = edited
        restoration = (
            "no undo package was present; the supplied checkpoint was used as-is"
        )
    del edited
    passage = json.loads(PASSAGE.read_text())
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    target = tokenizer.encode(passage["target_text"])
    collateral_ids = tokenizer.encode(passage["collateral_text"])
    prompt = 16
    dev = token_windows(target, prompt, 16)
    heldout = token_windows(target[16 * 17 :], prompt, 16)
    collateral = token_windows(collateral_ids, prompt, 8)
    seeds_report = []
    for seed in range(seeds):
        report = compare(model, dev, heldout, collateral, seed=seed, trials=trials)
        seeds_report.append({"seed": seed, "methods": compact(report)})
    return {
        "family": model.config.model_type,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "fingerprint": fingerprint(model),
        "restoration": restoration,
        "passage": {
            "work": passage["source"]["work"],
            "author": passage["source"]["author"],
            "rights": passage["source"]["rights"],
            "url": passage["source"]["url"],
            "sha256": passage["sha256"],
            "note": passage["source"]["note"],
        },
        "windows": {
            "prompt_tokens": prompt,
            "development": 16,
            "heldout": 16,
            "collateral": 8,
        },
        "seeds": seeds_report,
        "methods_better_than_untouched_on_every_seed": consistent(
            [row["methods"] for row in seeds_report]
        ),
        "interpretation": (
            "Next-token windows from a public-domain novel. The same windows are "
            "repeated across seeds; only the random arm draws new proposals. "
            "Collateral is a later chapter and is not used for selection. This is "
            "not a language-benchmark win and is separate from symbolic rule rewriting."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", default=os.environ.get("WEIGHTSCRIPT_UPSTREAM"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--language-checkpoint")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["bin_majority", "unique_copy", "sort"],
    )
    args = parser.parse_args()
    if not args.checkout:
        raise SystemExit("pass --checkout or set WEIGHTSCRIPT_UPSTREAM")
    torch.set_num_threads(2)
    checkout = Path(args.checkout)
    report = {
        "protocol": (
            "Three seeds. Each selective arm evaluates the same number of proposals "
            "on development data only. Untouched, random, scaling, direction, direct, "
            "program and hybrid arms are included. Direct and program apply one "
            "head-scale grid. No final-set failures are used to choose edits."
        ),
        "upstream_revision": revision(checkout),
        "seeds": args.seeds,
        "trials": args.trials,
        "tasks": {},
        "language_model": None,
        "grok_agent": {
            "status": "not_run_in_this_command",
            "reason": "baseline arms do not call an external agent",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        text = json.dumps(report, indent=2)
        if "/root/" in text or "/dev/shm" in text:
            raise SystemExit("refusing to write a report that contains a machine path")
        output.write_text(text + "\n")

    for task in args.tasks:
        directory = checkout / "share" / "saved_models" / TASKS[task]["checkpoint"]
        model = load_model(directory)
        report["tasks"][task] = run_task(model, task, args.seeds, args.trials)
        del model
        gc.collect()
        save()
    if args.language_checkpoint:
        report["language_model"] = run_language(
            args.language_checkpoint, args.seeds, args.trials
        )
        gc.collect()
        save()
    save()
    print(
        json.dumps(
            {
                task: row["methods_better_than_untouched_on_every_seed"]
                for task, row in report["tasks"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
