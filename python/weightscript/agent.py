"""Audited agent tool loop, shared by direct, program and hybrid editing.

The policy is an external JSON-lines command. It is trusted code, not a sandbox;
only inference requested through these tools is metered. Held-out rows are never
sent to the policy. Reports explicitly scope that guarantee.
"""

import copy
import json
import os
import signal
import subprocess
import tempfile
import time
import uuid
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
from .session_state import SessionState, exclusive_output
from .storage import atomic_json, digest


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


@exclusive_output
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
    resume=False,
    save_checkpoint=True,
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
    configuration = {
        "source_sha256": basehash,
        "command": list(command),
        "goal": goal,
        "mode": mode,
        "max_forwards": max_forwards,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "timeout": timeout,
        "seed": seed,
        "dev": digest(dev),
        "heldout": digest(heldout),
        "collateral": digest(collateral),
        "save_checkpoint": save_checkpoint,
    }
    state = SessionState(output, configuration, resume)
    data = state.data
    if data["completed"]:
        return json.loads((Path(output) / "report.json").read_text())
    ledger = data["history"]
    spent = data["spent"]
    cost = sum(1 for _ in batches(dev, batch_size))

    def reserve(n):
        nonlocal spent
        if spent + n > max_forwards:
            raise BudgetExceeded("inference budget exhausted")
        spent += n
        data["spent"] = spent
        state.save()

    if output and not resume:
        atomic_json(
            Path(output) / "protocol.json",
            dict(
                configuration,
                run_id=data["run_id"],
                runtime=provenance,
                claim="run configuration recorded before candidate evaluation; not external preregistration",
            ),
        )
    if data["baseline"] is None:
        reserve(cost)
        data["baseline"] = score(model, dev, batch_size)
        state.save()
    baseline = data["baseline"]
    best_metrics = data["best_metrics"] or baseline
    best_key = (best_metrics["accuracy"], -best_metrics["target_nll"])
    best_proposal = data["best_proposal"]
    best_model = state.restore(model, data["best_snapshot"])
    candidate = (
        best_model
        if data["candidate_snapshot"] == data["best_snapshot"]
        else state.restore(model, data["candidate_snapshot"])
    )
    graph = graph_from_model(model).to_dict()
    try:
        program = extract(model)
    except ValueError:
        program = None
    context = {
        "protocol": "weightscript.agent.v1",
        "run_id": data["run_id"],
        "goal": goal,
        "mode": mode,
        "seed": seed,
        "checkpoint_sha256": basehash,
        "architecture": {
            key: Path(value).name
            if isinstance(value, str) and value.startswith("/")
            else value
            for key, value in model.config.to_dict().items()
        },
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
    for step in range(
        data["next_step"] if not data["selection_complete"] else max_steps, max_steps
    ):
        request = data["pending_request"] or {
            "context": context,
            "step": step,
            "history": ledger,
            "remaining_forwards": max_forwards - spent,
            "remaining_steps": max_steps - step,
            "run_id": data["run_id"],
            "request_nonce": uuid.uuid4().hex,
        }
        data["pending_request"] = request
        state.save()
        began = time.monotonic()
        response = None
        try:
            response = data["pending_response"]
            if response is None:
                response = propose(command, request, timeout)
                if not isinstance(response, dict):
                    raise ValueError("policy response must be an object")
                echoed = response.get("request_nonce", request["request_nonce"])
                if (
                    echoed != request["request_nonce"]
                    or response.get("run_id", data["run_id"]) != data["run_id"]
                ):
                    raise ValueError(
                        "policy response nonce does not match this run request"
                    )
                data["pending_response"] = response
                state.save()
            elif (
                response.get("request_nonce", request["request_nonce"])
                != request["request_nonce"]
            ):
                raise ValueError("cached policy response nonce does not match")
            action = response.get("action")
            record = {"step": step, "response": response}
            if action == "finish":
                record["result"] = {"finished": True}
                record["request_nonce"] = request["request_nonce"]
                record["elapsed_seconds"] = time.monotonic() - began
                ledger.append(record)
                data.update(
                    next_step=step + 1,
                    pending_request=None,
                    pending_response=None,
                    selection_complete=True,
                )
                state.save()
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
                data["spent"] = spent
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
                data["candidate_snapshot"] = state.snapshot(model, proposed)
                key = (metrics["accuracy"], -metrics["target_nll"])
                if key > best_key:
                    best_key = key
                    best_model = proposed
                    best_proposal = response["proposal"]
                    best_metrics = metrics
                    data.update(
                        best_metrics=metrics,
                        best_proposal=best_proposal,
                        best_snapshot=data["candidate_snapshot"],
                    )
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
            data.update(
                selection_complete=True,
                pending_request=None,
                pending_response=None,
                next_step=step + 1,
            )
            state.save()
            break
        except (ValueError, KeyError, TypeError, RuntimeError, IndexError) as error:
            record = {
                "step": step,
                "error": str(error),
                "response": locals().get("response"),
                "request_nonce": request.get("request_nonce"),
            }
        record["elapsed_seconds"] = time.monotonic() - began
        record["request_nonce"] = request["request_nonce"]
        ledger.append(record)
        data["spent"] = spent
        data.update(next_step=step + 1, pending_request=None, pending_response=None)
        state.save()
        if output:
            Path(output).mkdir(parents=True, exist_ok=True)
            write_json(Path(output) / "journal.json", ledger)
    data["selection_complete"] = True
    state.save()
    if fingerprint(model) != basehash:
        raise RuntimeError("agent mutated original checkpoint")
    report = {
        "protocol": "weightscript.agent.v1",
        "run_id": data["run_id"],
        "controller_nonce": state.nonce,
        "resumed": resume,
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

    def final_metric(key, factory):
        if key not in data["final_metrics"]:
            data["final_metrics"][key] = factory()
            state.save()
        return data["final_metrics"][key]

    if heldout:
        report["baseline_heldout"] = final_metric(
            "baseline_heldout", lambda: score(model, heldout, batch_size)
        )
        report["selected_heldout"] = final_metric(
            "selected_heldout",
            lambda: score(best_model, heldout, batch_size, reference=model),
        )
    if collateral:
        report["baseline_collateral"] = final_metric(
            "baseline_collateral", lambda: score(model, collateral, batch_size)
        )
        report["selected_collateral"] = final_metric(
            "selected_collateral",
            lambda: score(best_model, collateral, batch_size, reference=model),
        )
    if output:
        from .checkpoint import save_undo

        root = Path(output)
        if data["best_snapshot"] is None:
            data["best_snapshot"] = state.snapshot(model, best_model)
            state.save()
        report["delta_checkpoint"] = data["best_snapshot"]
        if save_checkpoint:
            checkpoint = root / "checkpoint"
            if not checkpoint.exists():
                temporary = root / (".checkpoint-" + uuid.uuid4().hex)
                best_model.save_pretrained(temporary)
                save_undo(model, best_model, temporary)
                temporary.rename(checkpoint)
            else:
                from .core import load_model

                if fingerprint(load_model(checkpoint)) != fingerprint(best_model):
                    raise ValueError(
                        "existing exported checkpoint differs from selected candidate"
                    )
        atomic_json(root / "report.json", report)
    data["completed"] = True
    state.save()
    return report
