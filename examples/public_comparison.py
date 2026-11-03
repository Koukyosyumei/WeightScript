"""Public-data editing comparison for the pinned upstream tasks and one small LM.

Development rows select each arm. Held-out and collateral rows are scored once,
over every target token. Direct and program arms use the same head-scale grid.
This script does not declare a winning method from a small accuracy or NLL
difference.

The repeated-window language result is docs/results/public-baselines.json and
must be kept. This command writes a new file. Point --output somewhere else.
"""

import argparse
import gc
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch

from weightscript.comparison import (
    SPAN_GRADER,
    TASKS,
    compare,
    disjoint_seed_windows,
    task_rows,
)
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
            "development_labeled": row["development"]["labeled"],
            "development_min_answer_tokens": row["development"]["min_answer_tokens"],
            "collateral_disagreement": row["collateral"]["token_disagreement"],
            "collateral_kl": row["collateral"]["reference_kl_per_token"],
            "collateral_labeled": row["collateral"]["labeled"],
            "development_forward_calls": row["development_forward_calls"],
            "heldout_forward_calls": row["heldout"]["forward_calls"],
            "heldout_labeled": row["heldout"]["labeled"],
            "heldout_min_answer_tokens": row["heldout"]["min_answer_tokens"],
            "collateral_forward_calls": row["collateral"]["forward_calls"],
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


def span_counts(rows):
    lengths = [len(row["targets"]) for row in rows]
    return {
        "examples": len(rows),
        "labeled_positions": sum(lengths),
        "min_answer_tokens": min(lengths),
        "max_answer_tokens": max(lengths),
    }


def run_task(model, task, seeds, trials):
    rows = []
    for seed in range(seeds):
        dev, heldout, collateral = splits(task, seed)
        report = compare(model, dev, heldout, collateral, seed=seed, trials=trials)
        rows.append(
            {
                "seed": seed,
                "spans": {
                    "development": span_counts(dev),
                    "heldout": span_counts(heldout),
                    "collateral": span_counts(collateral),
                },
                "methods": compact(report),
            }
        )
    return {
        "provenance": {
            key: TASKS[task][key]
            for key in ("repository", "revision", "source", "checkpoint")
        },
        "grader": SPAN_GRADER,
        "seeds": rows,
        "methods_better_than_untouched_on_every_seed": consistent(
            [row["methods"] for row in rows]
        ),
        "interpretation": (
            "Held-out accuracy and NLL must both improve on every seed before a "
            "method is listed. Every target token is scored. A first-token match "
            "is not a full-span success. A single-seed or NLL-only change is not "
            "superiority."
        ),
    }


def _bounds(rows):
    indexes = [row["window_index"] for row in rows]
    return {"start": indexes[0], "stop": indexes[-1], "count": len(indexes)}


def _unique_windows(groups):
    seen = [row["window_index"] for rows in groups for row in rows]
    if len(seen) != len(set(seen)):
        raise SystemExit("language windows repeat across seeds")


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
    legacy = hashlib.sha256(
        (passage["target_text"] + "\n" + passage["collateral_text"]).encode()
    ).hexdigest()
    if legacy != passage["sha256"]:
        raise SystemExit("preserved passage sha256 does not match its text")
    block = passage["disjoint_evaluation"]
    body = "\n".join(
        [
            block["development_text"],
            block["heldout_text"],
            block["collateral_text"],
        ]
    )
    digest = hashlib.sha256(body.encode()).hexdigest()
    if digest != block["sha256"]:
        raise SystemExit("disjoint passage sha256 does not match its text")
    smoke = json.loads(
        (
            PASSAGE.resolve().parents[2] / "docs" / "results" / "distilgpt2-smoke.json"
        ).read_text()
    )
    undo_doc = (
        json.loads((path / "undo.json").read_text())
        if (path / "undo.json").is_file()
        else {}
    )
    restored = fingerprint(model)
    if undo_doc and restored != undo_doc["before_sha256"]:
        raise SystemExit("restored fingerprint does not match the undo package")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    tokenizer.model_max_length = 10**9
    prompt = 16
    dev_count, held_count, collateral_count = 8, 8, 4
    seed_ids = list(range(seeds))
    encoded = {
        name: tokenizer.encode(block[name], add_special_tokens=False, truncation=False)
        for name in ("development_text", "heldout_text", "collateral_text")
    }
    del tokenizer
    dev_by_seed = disjoint_seed_windows(
        encoded["development_text"], prompt, seed_ids, dev_count
    )
    held_by_seed = disjoint_seed_windows(
        encoded["heldout_text"], prompt, seed_ids, held_count
    )
    collateral_by_seed = disjoint_seed_windows(
        encoded["collateral_text"], prompt, seed_ids, collateral_count
    )
    _unique_windows(dev_by_seed.values())
    _unique_windows(held_by_seed.values())
    _unique_windows(collateral_by_seed.values())
    seeds_report = []
    forward_calls = 0
    for seed in seed_ids:
        report = compare(
            model,
            dev_by_seed[seed],
            held_by_seed[seed],
            collateral_by_seed[seed],
            seed=seed,
            trials=trials,
        )
        forward_calls += sum(
            row["development_forward_calls"]
            + row["heldout"]["forward_calls"]
            + row["collateral"]["forward_calls"]
            for row in report["methods"]
        )
        seeds_report.append(
            {
                "seed": seed,
                "windows": {
                    "development": _bounds(dev_by_seed[seed]),
                    "heldout": _bounds(held_by_seed[seed]),
                    "collateral": _bounds(collateral_by_seed[seed]),
                },
                "methods": compact(report),
            }
        )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if undo_doc and parameters != smoke["parameters"]:
        raise SystemExit("restored parameter count does not match the smoke report")
    identity = (
        (
            "The restored fingerprint equals the undo package before_sha256. "
            "The smoke report names this model and revision and records that undo "
            "restored the original hash. This run did not re-download the hub snapshot."
        )
        if undo_doc
        else restoration
    )
    return {
        "family": model.config.model_type,
        "model_name": smoke["model"],
        "upstream_revision": smoke["upstream_revision"],
        "parameters": parameters,
        "fingerprint": restored,
        "undo_before_sha256": undo_doc.get("before_sha256"),
        "smoke_undo_restored_original_hash": smoke["undo_restored_original_hash"],
        "restoration": restoration,
        "identity": identity,
        "passage": {
            "work": passage["source"]["work"],
            "author": passage["source"]["author"],
            "rights": passage["source"]["rights"],
            "url": passage["source"]["url"],
            "preserved_repeated_window_sha256": passage["sha256"],
            "disjoint_sha256": digest,
            "development_chapter": block["development_chapter"],
            "heldout_chapter": block["heldout_chapter"],
            "collateral_chapter": block["collateral_chapter"],
            "note": block["note"],
        },
        "windows": {
            "prompt_tokens": prompt,
            "development_per_seed": dev_count,
            "heldout_per_seed": held_count,
            "collateral_per_seed": collateral_count,
            "repeated_across_seeds": False,
            "answer_tokens": 1,
        },
        "grader": SPAN_GRADER,
        "seeds": seeds_report,
        "forward_calls": forward_calls,
        "api_cost_usd": 0,
        "methods_better_than_untouched_on_every_seed": consistent(
            [row["methods"] for row in seeds_report]
        ),
        "interpretation": (
            "One-target next-token windows from three disjoint public-domain "
            "chapters. Each seed receives a different slice. Development selects; "
            "held-out and collateral are scored once. An NLL-only change is not a "
            "capability gain. This is separate from the repeated-window result and "
            "from symbolic rule rewriting."
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
    if output.name in {"public-baselines.json", "grok-majority-pilot.json"}:
        raise SystemExit(f"refusing to overwrite preserved evidence {output.name}")
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
