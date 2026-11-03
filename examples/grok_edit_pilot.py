"""One-seed Grok editing pilot on the trained majority checkpoint.

The policy is the installed Grok CLI through weightscript.grok_exec. Held-out
and collateral rows are scored after the session stops. This script does not
invent a reply if the CLI fails.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from weightscript.agent import run_session
from weightscript.comparison import TASKS, task_rows
from weightscript.core import load_model


def labeled(rows):
    converted = []
    for row in rows:
        targets = row["targets"]
        if len(targets) != 1:
            raise ValueError("this pilot scores the single majority answer token")
        converted.append({"input_ids": row["input_ids"], "target": targets[0]})
    return converted


def splits(seed):
    return (
        labeled(task_rows("bin_majority", seed, [5, 7], 8)),
        labeled(task_rows("bin_majority", seed + 100, [11, 13], 8)),
        labeled(task_rows("bin_majority", seed + 200, [4], 6)),
    )


def scrub(report, revision):
    history = []
    for row in report["history"]:
        response = row.get("response") if isinstance(row.get("response"), dict) else {}
        result = row.get("result") if isinstance(row.get("result"), dict) else {}
        dev = result.get("dev") if isinstance(result.get("dev"), dict) else {}
        audit = result.get("audit") if isinstance(result.get("audit"), dict) else {}
        history.append(
            {
                "step": row.get("step"),
                "action": response.get("action"),
                "error": row.get("error"),
                "elapsed_seconds": row.get("elapsed_seconds"),
                "usage": response.get("usage"),
                "proposal": response.get("proposal"),
                "selected_best": result.get("selected_best"),
                "development_accuracy": dev.get("accuracy"),
                "development_nll": dev.get("target_nll"),
                "changed_scalars": audit.get("changed_scalars"),
            }
        )
    public = {
        "task": "bin_majority",
        "checkpoint": TASKS["bin_majority"]["checkpoint"],
        "repository": TASKS["bin_majority"]["repository"],
        "upstream_revision": revision,
        "seed": report["seed"],
        "mode": report["mode"],
        "goal": report["goal"],
        "run_id": report["run_id"],
        "budget": report["budget"],
        "invalid_steps": report["invalid_steps"],
        "baseline_development": report["baseline_dev"],
        "selected_development": report["selected_dev"],
        "selected_proposal": report["selected_proposal"],
        "baseline_heldout": report.get("baseline_heldout"),
        "selected_heldout": report.get("selected_heldout"),
        "baseline_collateral": report.get("baseline_collateral"),
        "selected_collateral": report.get("selected_collateral"),
        "changed_scalars": report["audit"]["changed_scalars"],
        "heldout_sent_to_policy": report["heldout_sent_to_policy"],
        "policy_usage": report["policy_usage"],
        "history": history,
        "elapsed_seconds": report["elapsed_seconds"],
        "metering_scope": report["metering_scope"],
        "interpretation": (
            "One seed and one trained majority checkpoint. The policy is a live "
            "Grok CLI turn, and usage fields are copied from that turn's JSON. "
            "Held-out rows were not sent to the policy. A development NLL change "
            "on this already accurate task is not a superiority result."
        ),
    }
    text = json.dumps(public)
    if "/root/" in text or "/dev/shm" in text:
        raise SystemExit("refusing to write a report that contains a machine path")
    return public


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-forwards", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    checkout = Path(args.checkout)
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    expected = TASKS["bin_majority"]["revision"]
    if revision != expected:
        raise SystemExit(f"expected upstream revision {expected}, got {revision}")
    checkpoint = (
        checkout / "share" / "saved_models" / TASKS["bin_majority"]["checkpoint"]
    )
    model = load_model(checkpoint)
    dev, heldout, collateral = splits(args.seed)
    goal = (
        "Improve development next-token accuracy and negative log likelihood for "
        "this binary-majority model. Select only on the development rows in the "
        "request. Inspect at most one tensor, evaluate at most one proposal, then "
        "finish. Do not ask for held-out labels."
    )
    with tempfile.TemporaryDirectory(prefix="weightscript-grok-pilot-") as temporary:
        workspace = Path(temporary) / "workspace"
        workspace.mkdir()
        command = [
            sys.executable,
            "-m",
            "weightscript.grok_exec",
            "--workspace",
            str(workspace),
            "--timeout",
            str(args.timeout),
        ]
        report = run_session(
            model,
            command,
            goal,
            dev,
            heldout=heldout,
            collateral=collateral,
            mode="program",
            max_forwards=args.max_forwards,
            max_steps=args.max_steps,
            timeout=args.timeout,
            seed=args.seed,
            output=str(Path(temporary) / "session"),
            save_checkpoint=False,
        )
    public = scrub(report, revision)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(public, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(destination),
                "invalid_steps": public["invalid_steps"],
                "actions": [row["action"] for row in public["history"]],
                "policy_usage": public["policy_usage"],
                "changed_scalars": public["changed_scalars"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
