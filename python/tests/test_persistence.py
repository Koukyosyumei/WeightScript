import copy
import json
import sys

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from weightscript import agent
from weightscript.core import fingerprint
from weightscript.jobs import JobStore
from weightscript.patches import apply_delta, save_delta
from weightscript.storage import RunBusyError, atomic_json, controller_lock


@pytest.fixture
def model():
    torch.manual_seed(24)
    torch.set_num_threads(1)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=13,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=2,
            _attn_implementation="eager",
        )
    ).eval()


def test_delta_is_source_bound_alias_preserving_and_small(model, tmp_path):
    original = fingerprint(model)
    modified = copy.deepcopy(model)
    with torch.no_grad():
        modified.lm_head.weight[2, 1] += 1
    save_delta(model, modified, tmp_path / "delta")
    recovered = apply_delta(model, tmp_path / "delta")
    assert fingerprint(recovered) == fingerprint(modified)
    assert (
        recovered.lm_head.weight.data_ptr()
        == recovered.transformer.wte.weight.data_ptr()
    )
    assert fingerprint(model) == original
    with pytest.raises(ValueError, match="source"):
        apply_delta(modified, tmp_path / "delta")
    with pytest.raises(ValueError):
        save_delta(model, modified, tmp_path / "delta")


def test_atomic_writes_and_single_controller(tmp_path):
    path = tmp_path / "state.json"
    atomic_json(path, {"complete": True})
    with pytest.raises(ValueError):
        atomic_json(path, {"bad": float("nan")})
    assert json.loads(path.read_text()) == {"complete": True}
    with controller_lock(tmp_path):
        with pytest.raises(RunBusyError):
            with controller_lock(tmp_path):
                pass
    with controller_lock(tmp_path):
        pass


def test_policy_context_uses_checkpoint_basename(model, tmp_path, monkeypatch):
    model.config._name_or_path = "/tmp/secret-checkpoint/weights"
    seen = {}

    def policy(command, request, timeout):
        seen["architecture"] = request["context"]["architecture"]
        return {"action": "finish"}

    monkeypatch.setattr(agent, "propose", policy)
    agent.run_session(
        model,
        ["policy"],
        "goal",
        [{"input_ids": [1, 2, 3], "target": 4}],
        output=tmp_path / "session",
        max_steps=1,
        max_forwards=4,
        save_checkpoint=False,
    )
    assert seen["architecture"]["_name_or_path"] == "weights"
    saved = "\n".join(
        path.read_text() for path in (tmp_path / "session").rglob("*.json")
    )
    assert "/tmp/" not in saved


def test_session_recovers_after_controller_interruption(model, tmp_path, monkeypatch):
    calls = []

    def policy(command, request, timeout):
        calls.append(request["step"])
        if request["step"] == 1:
            raise KeyboardInterrupt()
        return {
            "action": "evaluate",
            "proposal": {
                "edits": [{"op": "head_scale", "layer": 0, "head": 0, "factor": 0.5}]
            },
        }

    monkeypatch.setattr(agent, "propose", policy)
    rows = [{"input_ids": [1, 2, 3], "target": 4}]
    kwargs = dict(
        output=tmp_path / "session", max_steps=3, max_forwards=5, save_checkpoint=False
    )
    with pytest.raises(KeyboardInterrupt):
        agent.run_session(model, ["policy"], "goal", rows, **kwargs)
    state = json.loads((tmp_path / "session/state.json").read_text())
    assert state["next_step"] == 1 and state["spent"] == 2

    def finish(command, request, timeout):
        calls.append(request["step"])
        return {"action": "finish"}

    monkeypatch.setattr(agent, "propose", finish)
    result = agent.run_session(model, ["policy"], "goal", rows, resume=True, **kwargs)
    assert calls == [0, 1, 1]
    assert result["budget"]["selection_forwards"] == 2
    assert result["history"][0]["step"] == 0
    assert result["resumed"]
    # A completed resume performs no policy or model evaluation again.
    assert (
        agent.run_session(model, ["policy"], "goal", rows, resume=True, **kwargs)
        == result
    )
    with pytest.raises(ValueError, match="differs"):
        agent.run_session(
            model, ["policy"], "changed goal", rows, resume=True, **kwargs
        )


def test_cached_policy_response_survives_failed_evaluation(
    model, tmp_path, monkeypatch
):
    policy_calls = []
    original_score = agent.score_objective
    scoring = [0]

    def policy(*args):
        policy_calls.append(1)
        return {"action": "evaluate", "proposal": {"edits": []}}

    def interrupt(*args, **kwargs):
        scoring[0] += 1
        if scoring[0] == 2:
            raise KeyboardInterrupt()
        return original_score(*args, **kwargs)

    monkeypatch.setattr(agent, "propose", policy)
    monkeypatch.setattr(agent, "score_objective", interrupt)
    rows = [{"input_ids": [1, 2], "target": 3}]
    config = dict(
        max_steps=1, max_forwards=4, output=tmp_path / "session", save_checkpoint=False
    )
    with pytest.raises(KeyboardInterrupt):
        agent.run_session(model, ["policy"], "goal", rows, **config)
    monkeypatch.setattr(agent, "score_objective", original_score)
    result = agent.run_session(model, ["policy"], "goal", rows, resume=True, **config)
    assert len(policy_calls) == 1
    assert (
        result["budget"]["selection_forwards"] == 3
    )  # interrupted reservation remains charged


def test_job_metadata_and_explicit_command_execution(tmp_path):
    with JobStore(tmp_path / "jobs") as jobs:
        key = jobs.create([sys.executable, "-c", 'print("durable")'])
        assert jobs.get(key)["status"] == "pending"
        assert jobs.execute(key) == 0
        assert jobs.get(key)["status"] == "succeeded"
        assert jobs.logs(key).strip() == "durable"
    with JobStore(tmp_path / "jobs") as jobs:
        assert jobs.get(key)["attempts"] == 1
        assert jobs.execute(key) == 0
        assert jobs.get(key)["attempts"] == 1


def test_dense_and_partial_deltas_fail_closed(model, tmp_path):
    modified = copy.deepcopy(model)
    with torch.no_grad():
        modified.transformer.h[0].attn.c_proj.weight.add_(0.25)
    manifest = save_delta(model, modified, tmp_path / "dense")
    assert manifest["tensors"] and manifest["tensors"][0]["sparse"] is False
    recovered = apply_delta(model, tmp_path / "dense")
    assert fingerprint(recovered) == fingerprint(modified)
    (tmp_path / "dense" / "delta.safetensors").unlink()
    with pytest.raises(ValueError, match="missing"):
        apply_delta(model, tmp_path / "dense")
    assert fingerprint(model) == manifest["source_sha256"]
    save_delta(model, copy.deepcopy(model), tmp_path / "empty")
    assert fingerprint(apply_delta(model, tmp_path / "empty")) == fingerprint(model)


def test_sparse_snapshot_survives_interruption(model, tmp_path, monkeypatch):
    calls = []

    def policy(command, request, timeout):
        calls.append(request["request_nonce"])
        if request["step"] == 0:
            return {
                "action": "evaluate",
                "proposal": {
                    "transaction": [
                        {
                            "op": "add",
                            "tensor": "transformer.h.0.attn.c_attn.weight",
                            "select": [[1], [2]],
                            "value": 0.5,
                        }
                    ]
                },
                "request_nonce": request["request_nonce"],
                "run_id": request["run_id"],
            }
        raise KeyboardInterrupt()

    monkeypatch.setattr(agent, "propose", policy)
    rows = [{"input_ids": [1, 2, 3], "target": 4}]
    output = tmp_path / "session"
    kwargs = dict(output=output, max_steps=2, max_forwards=6, save_checkpoint=False)
    with pytest.raises(KeyboardInterrupt):
        agent.run_session(model, ["policy"], "goal", rows, mode="direct", **kwargs)
    manifests = list((output / "snapshots").glob("*/manifest.json"))
    manifest = json.loads(manifests[0].read_text())
    assert manifest["tensors"][0]["sparse"]
    assert (
        fingerprint(apply_delta(model, manifests[0].parent))
        == manifest["result_sha256"]
    )
    before = json.loads((output / "state.json").read_text())

    def finish(command, request, timeout):
        calls.append(request["request_nonce"])
        return {"action": "finish", "request_nonce": request["request_nonce"]}

    monkeypatch.setattr(agent, "propose", finish)
    result = agent.run_session(
        model, ["policy"], "goal", rows, mode="direct", resume=True, **kwargs
    )
    assert calls[1] == calls[2]
    assert result["history"][0]["result"]["audit"]["changed_scalars"] == 1
    assert (
        fingerprint(apply_delta(model, manifests[0].parent))
        == manifest["result_sha256"]
    )
    assert result["controller_nonce"] != before["controllers"][0]["nonce"]
    assert result["run_id"] == before["run_id"]


def test_mismatched_policy_nonce_is_not_cached(model, tmp_path, monkeypatch):
    def policy(command, request, timeout):
        return {
            "action": "finish",
            "request_nonce": "stale",
            "run_id": request["run_id"],
        }

    monkeypatch.setattr(agent, "propose", policy)
    rows = [{"input_ids": [1], "target": 2}]
    result = agent.run_session(
        model,
        ["policy"],
        "goal",
        rows,
        mode="direct",
        max_steps=1,
        max_forwards=2,
        output=tmp_path / "session",
        save_checkpoint=False,
    )
    assert "nonce" in result["history"][0]["error"]
    state = json.loads((tmp_path / "session" / "state.json").read_text())
    assert state["pending_response"] is None


def test_generation_refund_remains_after_resume(model, tmp_path, monkeypatch):
    def fake_generate(model, input_ids, max_new_tokens=32, temperature=0.0, seed=0):
        return {
            "tokens": [1],
            "forward_calls": 1,
            "stop_reason": "eos",
            "repeated_4gram_fraction": 0.0,
        }

    def policy(command, request, timeout):
        if request["step"] == 0:
            return {"action": "generate", "input_ids": [1, 2], "max_new_tokens": 3}
        raise KeyboardInterrupt()

    monkeypatch.setattr(agent, "propose", policy)
    monkeypatch.setattr("weightscript.generation.generate", fake_generate)
    rows = [{"input_ids": [1, 2], "target": 3}]
    output = tmp_path / "session"
    kwargs = dict(output=output, max_steps=2, max_forwards=8, save_checkpoint=False)
    with pytest.raises(KeyboardInterrupt):
        agent.run_session(model, ["policy"], "goal", rows, mode="direct", **kwargs)
    # Baseline costs one forward. Generation reserved 3 and refunded 2.
    assert json.loads((output / "state.json").read_text())["spent"] == 2


def test_sequence_session_scores_every_answer_position(model, tmp_path, monkeypatch):
    from types import SimpleNamespace

    def scripted(input_ids, use_cache=False, **kwargs):
        batch, seq = input_ids.shape
        logits = torch.full((batch, seq, model.config.vocab_size), -20.0)
        logits[:, 3, 5] = 10
        logits[:, 4, 1] = 10
        return SimpleNamespace(logits=logits)

    model.forward = scripted
    seen = {}

    def policy(command, request, timeout):
        seen["grader"] = request["context"]["grader"]
        return {"action": "finish"}

    monkeypatch.setattr(agent, "propose", policy)
    report = agent.run_session(
        model,
        ["policy"],
        "goal",
        [{"input_ids": [1, 2, 3, 4], "targets": [5, 6], "answer_length": 2}],
        heldout=[{"input_ids": [1, 2, 3, 0], "targets": [5, 6], "answer_length": 2}],
        collateral=[{"input_ids": [1, 2, 0, 4], "targets": [5, 7], "answer_length": 2}],
        objective="sequence",
        output=tmp_path / "span",
        max_steps=1,
        max_forwards=4,
        save_checkpoint=False,
    )
    assert "every target" in seen["grader"]
    assert report["objective"] == "sequence"
    assert report["baseline_dev"]["examples"] == 1
    assert report["baseline_dev"]["labeled"] == 2
    assert report["baseline_dev"]["accuracy"] == 0.5
    assert report["baseline_dev"]["full_span"] is True
    assert report["baseline_heldout"]["accuracy"] == 0.5
    assert report["baseline_heldout"]["labeled"] == 2
    assert report["selected_collateral"]["labeled"] == 2
    assert report["budget"]["selection_forwards"] == 1


def test_truncated_copy_does_not_call_the_policy(model, tmp_path, monkeypatch):
    from weightscript.comparison import task_rows

    called = []
    monkeypatch.setattr(
        agent,
        "propose",
        lambda *args, **kwargs: called.append(1) or {"action": "finish"},
    )
    full = task_rows("unique_copy", 0, [4], 1)[0]
    short = {
        "input_ids": full["input_ids"],
        "targets": full["targets"][:1],
        "task": "unique_copy",
        "answer_length": 1,
    }
    with pytest.raises(ValueError, match="truncated"):
        agent.run_session(
            model,
            ["policy"],
            "goal",
            [short],
            objective="sequence",
            output=tmp_path / "bad",
            max_steps=1,
            max_forwards=4,
            save_checkpoint=False,
        )
    assert called == []


def test_sequence_budget_uses_prepared_length(model, tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "propose", lambda *args, **kwargs: {"action": "finish"})
    rows = [
        {"input_ids": [1, 2, 3], "targets": [4, 5], "answer_length": 2},
        {"input_ids": [1, 2], "targets": [3, 4, 5], "answer_length": 3},
        {"input_ids": [1, 2, 3, 4], "targets": [5, 6, 7], "answer_length": 3},
    ]
    report = agent.run_session(
        model,
        ["policy"],
        "goal",
        rows,
        objective="sequence",
        output=tmp_path / "budget",
        max_steps=1,
        max_forwards=2,
        save_checkpoint=False,
    )
    assert report["budget"]["selection_forwards"] == 2
    assert report["baseline_dev"]["labeled"] == 8
    assert report["baseline_dev"]["labeled"] > report["baseline_dev"]["examples"]
    overlap = rows[0]
    called = []
    monkeypatch.setattr(agent, "propose", lambda *args, **kwargs: called.append(1))
    with pytest.raises(ValueError, match="collateral"):
        agent.run_session(
            model,
            ["policy"],
            "goal",
            [overlap],
            collateral=[overlap],
            objective="sequence",
            max_steps=1,
            max_forwards=2,
            save_checkpoint=False,
        )
    assert called == []


def test_lock_is_released_when_the_holder_is_killed(tmp_path):
    import os
    import subprocess

    script = """
import os, time
from weightscript.storage import controller_lock
with controller_lock(os.environ["ROOT"]):
    print("locked", flush=True)
    time.sleep(30)
"""
    env = dict(os.environ, ROOT=str(tmp_path))
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout.readline().strip() == "locked"
    proc.kill()
    proc.wait(timeout=5)
    with controller_lock(tmp_path):
        pass
