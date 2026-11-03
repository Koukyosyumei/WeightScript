"""Audited agent tool loop, shared by direct, program and hybrid editing.

The policy is an external JSON-lines command. It is trusted code, not a sandbox;
only inference requested through these tools is metered. Held-out rows are never
sent to the policy. Reports explicitly scope that guarantee.
"""

import copy
import json
import hashlib
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import torch

from .core import (
    apply_raw_edits,
    audit,
    compile_edits,
    extract,
    fingerprint,
    program_forward,
    write_json,
)
from .evaluate import batches, score
from .ir import graph_from_model
from .raw import transaction


class BudgetExceeded(ValueError):
    pass


def propose(command, request, timeout=60):
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            process.communicate(
                json.dumps(request, allow_nan=False).encode(), timeout=timeout
            )
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise ValueError("agent policy command timed out")
        if stdout.tell() > 1024 * 1024 or stderr.tell() > 1024 * 1024:
            raise ValueError("agent response exceeds 1 MiB")
        stdout.seek(0)
        stderr.seek(0)
        if process.returncode:
            raise ValueError(
                f"agent command failed: {stderr.read().decode(errors='replace')[:2000]}"
            )
        return json.loads(stdout.read())


def build_candidate(model, proposal, mode, rule_domain=None):
    if not isinstance(proposal, dict):
        raise ValueError("proposal must be an object")
    allowed = (
        {"edits", "rule"}
        if mode == "program"
        else {"raw", "transaction"}
        if mode == "direct"
        else {"edits", "rule", "raw", "transaction"}
    )
    if not proposal or set(proposal) - allowed:
        raise ValueError(f"proposal violates {mode} editing interface")
    result = model
    if "rule" in proposal:
        from .synthesis import compile_rule

        if rule_domain is None:
            raise ValueError("rule compilation needs an explicit synthesis domain")
        result, _ = compile_rule(model, proposal["rule"], rule_domain, strict=True)
    if "edits" in proposal:
        result = compile_edits(result, dict(extract(result), edits=proposal["edits"]))
    if "raw" in proposal:
        result = apply_raw_edits(result, proposal["raw"])
    if "transaction" in proposal:
        result = transaction(result, proposal["transaction"])
    return copy.deepcopy(model).eval() if result is model else result


def run_session(
    model,
    command,
    goal,
    dev,
    heldout=None,
    mode="hybrid",
    max_forwards=50,
    max_steps=30,
    batch_size=32,
    timeout=60,
    seed=0,
    output=None,
    collateral=None,
):
    if (
        mode not in {"program", "direct", "hybrid"}
        or max_forwards < 1
        or max_steps < 1
        or batch_size < 1
    ):
        raise ValueError("invalid session configuration")
    if not dev or not all("target" in r for r in dev):
        raise ValueError("labeled development data required")
    if heldout and {tuple(r["input_ids"]) for r in dev} & {
        tuple(r["input_ids"]) for r in heldout
    }:
        raise ValueError("development and held-out data overlap")
    from .provenance import runtime

    provenance = runtime()
    torch.manual_seed(seed)
    basehash = fingerprint(model)
    start = time.monotonic()
    ledger = []
    spent = 0
    cost = sum(1 for _ in batches(dev, batch_size))

    def reserve(n):
        nonlocal spent
        if spent + n > max_forwards:
            raise BudgetExceeded("inference budget exhausted")
        spent += n

    if output:
        root = Path(output)
        if root.exists() and any(root.iterdir()):
            raise ValueError("session output must be empty")
        root.mkdir(parents=True, exist_ok=True)
        write_json(
            root / "protocol.json",
            {
                "mode": mode,
                "seed": seed,
                "base_sha256": basehash,
                "max_forwards": max_forwards,
                "max_steps": max_steps,
                "command": list(command),
                "dev_sha256": hashlib.sha256(
                    json.dumps(dev, sort_keys=True).encode()
                ).hexdigest(),
                "heldout_sha256": hashlib.sha256(
                    json.dumps(heldout, sort_keys=True).encode()
                ).hexdigest(),
                "claim": "run configuration recorded before candidate evaluation; not external preregistration",
                "runtime": provenance,
            },
        )
    reserve(cost)
    baseline = score(model, dev, batch_size)
    best_key = (baseline["accuracy"], -baseline["target_nll"])
    best_model = model
    best_proposal = {}
    best_metrics = baseline
    candidate = model
    graph = graph_from_model(model).to_dict()
    try:
        program = extract(model)
    except ValueError:
        program = None
    context = {
        "protocol": "weightscript.agent.v1",
        "goal": goal,
        "mode": mode,
        "seed": seed,
        "checkpoint_sha256": basehash,
        "architecture": model.config.to_dict(),
        "weights": {name: list(w.shape) for name, w in model.state_dict().items()},
        "program": program if mode != "direct" else None,
        "graph": graph if mode != "direct" else None,
        "development_examples": dev,
        "baseline_dev": baseline,
        "actions": {
            "inspect": "tensor, optional start/stop rows and columns (<=32 each)",
            "trace": "input_ids; costs one model forward",
            "analyze_tensor": "tensor, components (1..8); returns weight norms and SVD directions; no inference",
            "evaluate": "proposal; edits are relative to original checkpoint",
            "generate": "input_ids, max_new_tokens; native generation on latest candidate, token budget charged",
            "finish": "stop; selected best development candidate is retained",
        },
        "proposal_interfaces": sorted(
            {"edits", "rule"}
            if mode == "program"
            else {"raw", "transaction"}
            if mode == "direct"
            else {"edits", "rule", "raw", "transaction"}
        ),
        "rule_schema": {
            "op": "count_threshold",
            "threshold": "number in [0,1]",
            "invert": "boolean",
        }
        if model.config.model_type == "gpt2"
        else None,
    }
    example_tensor = next(
        name
        for name, w in model.state_dict().items()
        if w.ndim == 2 and name != "lm_head.weight"
    )
    examples = [
        {
            "action": "inspect",
            "tensor": example_tensor,
            "start": 0,
            "stop": 2,
            "col_start": 0,
            "col_stop": 2,
        },
        {"action": "trace", "input_ids": dev[0]["input_ids"]},
        {"action": "finish"},
    ]
    if mode in {"direct", "hybrid"}:
        examples.append(
            {
                "action": "evaluate",
                "proposal": {
                    "transaction": [
                        {
                            "op": "scale",
                            "tensor": example_tensor,
                            "select": [None, None],
                            "value": 0.5,
                        }
                    ]
                },
            }
        )
        context["transaction_schema"] = {
            "op": ["set", "add", "scale", "low_rank_add"],
            "select": "one axis selector per dimension: null, explicit index list, or start/stop/step object",
            "value": "scalar or selected tensor values; low_rank_add uses left/right/factor instead of select/value",
        }
    if mode in {"program", "hybrid"} and program is not None:
        examples.append(
            {
                "action": "evaluate",
                "proposal": {
                    "edits": [
                        {"op": "head_scale", "layer": 0, "head": 0, "factor": 0.5}
                    ]
                },
            }
        )
        if model.config.model_type == "gpt2":
            examples.append(
                {
                    "action": "evaluate",
                    "proposal": {
                        "rule": {
                            "op": "count_threshold",
                            "threshold": 0.5,
                            "invert": False,
                        }
                    },
                }
            )
    context["action_examples"] = examples
    for step in range(max_steps):
        request = {
            "context": context,
            "step": step,
            "history": ledger,
            "remaining_forwards": max_forwards - spent,
            "remaining_steps": max_steps - step,
        }
        began = time.monotonic()
        response = None
        try:
            response = propose(command, request, timeout)
            action = response.get("action")
            record = {"step": step, "response": response}
            if action == "finish":
                record["result"] = {"finished": True}
                ledger.append(record)
                break
            if action == "inspect":
                name = response["tensor"]
                w = model.state_dict()[name]
                start = response.get("start", 0)
                stop = response.get("stop", min(start + 8, w.shape[0]))
                if (
                    type(start) is not int
                    or type(stop) is not int
                    or not 0 <= start < stop <= w.shape[0]
                    or stop - start > 32
                ):
                    raise ValueError("invalid inspection bounds")
                selected = w[start:stop]
                if w.ndim == 2:
                    begin = response.get("col_start", 0)
                    end = response.get("col_stop", min(begin + 8, w.shape[1]))
                    if (
                        type(begin) is not int
                        or type(end) is not int
                        or not 0 <= begin < end <= w.shape[1]
                        or end - begin > 32
                    ):
                        raise ValueError("invalid columns")
                    selected = selected[:, begin:end]
                elif w.ndim > 2:
                    raise ValueError("inspect vector/matrix tensors only")
                result = {"tensor": name, "values": selected.tolist()}
            elif action == "analyze_tensor":
                name = response["tensor"]
                weight = model.state_dict()[name]
                components = response.get("components", 3)
                if (
                    type(components) is not int
                    or not 1 <= components <= 8
                    or weight.ndim != 2
                    or weight.numel() > 4_000_000
                ):
                    raise ValueError(
                        "tensor analysis requires a matrix <=4M elements and 1..8 components"
                    )
                with torch.no_grad():
                    left, values, right = torch.linalg.svd(weight, full_matrices=False)
                result = {
                    "tensor": name,
                    "shape": list(weight.shape),
                    "singular_values": values[:components].tolist(),
                    "left_vectors": left[:, :components].T.tolist(),
                    "right_vectors": right[:components].tolist(),
                    "top_row_norms": torch.linalg.vector_norm(weight, dim=1)
                    .topk(min(16, weight.shape[0]))
                    .indices.tolist(),
                }
            elif action == "trace":
                ids = torch.tensor([response["input_ids"]], dtype=torch.long)
                if program is None:
                    raise ValueError("trace needs an architecture adapter")
                # Validate and charge before execution; selected candidate is explicit.
                reserve(1)
                with torch.inference_mode():
                    logits, traces = program_forward(
                        candidate, extract(candidate), ids, capture=True
                    )
                result = {"predictions": logits.argmax(-1).tolist(), "layers": []}
                for t in traces:
                    means = t["neurons"].abs().mean((0, 1))
                    top = means.topk(min(16, len(means)))
                    result["layers"].append(
                        {
                            "layer": t["layer"],
                            "attention": t["attention"].tolist(),
                            "top_neurons": top.indices.tolist(),
                            "activation": top.values.tolist(),
                        }
                    )
            elif action == "generate":
                from .generation import generate

                amount = response.get("max_new_tokens", 32)
                if type(amount) is not int or not 1 <= amount <= 256:
                    raise ValueError("invalid generation limit")
                reserve(amount)
                result = generate(candidate, response["input_ids"], amount, seed=seed)
                spent -= amount - result["forward_calls"]
            elif action == "evaluate":
                proposal = response["proposal"]
                # Reserve all compiler probes before invoking constraint synthesis.
                allowed = (
                    {"edits", "rule"}
                    if mode == "program"
                    else {"raw", "transaction"}
                    if mode == "direct"
                    else {"edits", "rule", "raw", "transaction"}
                )
                if not isinstance(proposal, dict) or set(proposal) - allowed:
                    raise ValueError(f"proposal violates {mode} editing interface")
                internal_cost = (
                    2 * sum(1 for _ in batches(dev, 128)) if "rule" in proposal else 0
                )
                reserve(cost + internal_cost)
                proposed = build_candidate(model, proposal, mode, dev)
                metrics = score(proposed, dev, batch_size)
                candidate = proposed
                key = (metrics["accuracy"], -metrics["target_nll"])
                if key > best_key:
                    best_key = key
                    best_model = proposed
                    best_proposal = response["proposal"]
                    best_metrics = metrics
                result = {
                    "dev": metrics,
                    "audit": audit(model, proposed),
                    "selected_best": key == best_key,
                }
            else:
                raise ValueError("unknown agent action")
            record["result"] = result
        except BudgetExceeded as error:
            ledger.append({"step": step, "error": str(error), "budget_exhausted": True})
            break
        except (ValueError, KeyError, TypeError, RuntimeError, IndexError) as error:
            record = {
                "step": step,
                "error": str(error),
                "response": locals().get("response"),
            }
        record["elapsed_seconds"] = time.monotonic() - began
        ledger.append(record)
        if output:
            Path(output).mkdir(parents=True, exist_ok=True)
            write_json(Path(output) / "journal.json", ledger)
    if fingerprint(model) != basehash:
        raise RuntimeError("agent mutated original checkpoint")
    report = {
        "protocol": "weightscript.agent.v1",
        "runtime": provenance,
        "mode": mode,
        "goal": goal,
        "seed": seed,
        "base_sha256": basehash,
        "policy_command": list(command),
        "budget": {
            "max_forwards": max_forwards,
            "selection_forwards": spent,
            "max_steps": max_steps,
            "steps": len(ledger),
            "policy_timeout_seconds": timeout,
        },
        "invalid_steps": sum("error" in row for row in ledger),
        "baseline_dev": baseline,
        "selected_dev": best_metrics,
        "selected_proposal": best_proposal,
        "audit": audit(model, best_model),
        "history": ledger,
        "elapsed_seconds": time.monotonic() - start,
        "metering_scope": "framework tool inference only; external policy is trusted and can access host resources; not a security sandbox",
        "heldout_sent_to_policy": False,
        "policy_usage": [
            (r.get("response") or {}).get("usage")
            for r in ledger
            if (r.get("response") or {}).get("usage")
        ],
    }
    if heldout:
        report["baseline_heldout"] = score(model, heldout, batch_size)
        report["selected_heldout"] = score(
            best_model, heldout, batch_size, reference=model
        )
    if collateral:
        report["baseline_collateral"] = score(model, collateral, batch_size)
        report["selected_collateral"] = score(
            best_model, collateral, batch_size, reference=model
        )
    if output:
        from .checkpoint import save_undo

        root = Path(output)
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "report.json", report)
        checkpoint = root / "checkpoint"
        if checkpoint.exists():
            raise ValueError("checkpoint output already exists")
        best_model.save_pretrained(checkpoint)
        save_undo(model, best_model, checkpoint)
    return report
