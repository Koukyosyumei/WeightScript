import sys

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from weightscript.core import audit, fingerprint
from weightscript.raw import transaction
from weightscript.ir import Artifact, Node, graph_from_model
from weightscript.agent import run_session


@pytest.fixture
def model():
    torch.set_num_threads(1)
    torch.manual_seed(12)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=13,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=2,
            attn_pdrop=0,
            resid_pdrop=0,
            embd_pdrop=0,
            _attn_implementation="eager",
        )
    ).eval()


def test_raw_tensor_transactions_are_atomic_and_alias_aware(model):
    before = fingerprint(model)
    edit = {
        "op": "set",
        "tensor": "lm_head.weight",
        "select": [[1, 3], [0, 2]],
        "value": [[1.0, 2.0], [3.0, 4.0]],
    }
    changed = transaction(model, [edit])
    assert changed.lm_head.weight[3, 2] == 4
    assert changed.transformer.wte.weight[1, 0] == 1
    assert audit(model, changed)["changed_scalars"] == 4
    assert fingerprint(model) == before
    invalid = dict(edit, value=[[float("nan")] * 2] * 2)
    with pytest.raises(ValueError):
        transaction(model, [edit, invalid])
    assert fingerprint(model) == before
    bad = dict(edit, select=[[-1], [0]])
    with pytest.raises(ValueError):
        transaction(model, [bad])
    low = {
        "op": "low_rank_add",
        "tensor": "transformer.h.0.attn.c_proj.weight",
        "left": [[1.0]] * 8,
        "right": [[1.0] * 8],
        "factor": 0.1,
    }
    assert audit(model, transaction(model, [low]))["changed_scalars"] == 64


def test_ir_retains_opaque_modules_and_rejects_invalid_graph(model):
    graph = graph_from_model(model)
    graph.validate()
    assert graph.nodes[-1].interpretation == "opaque"
    with pytest.raises(ValueError):
        Artifact(
            "x", [Node("a", "x", ["missing"])], "a", {}, {"claim": "test"}
        ).validate()


def test_agent_budget_holdout_and_mode_enforcement(model, tmp_path):
    worker = tmp_path / "policy.py"
    worker.write_text("""import json,sys
r=json.load(sys.stdin)
assert 'heldout' not in r['context']
if r['step']==0: a={'action':'evaluate','proposal':{'edits':[{'op':'head_scale','layer':0,'head':1,'factor':0.5}]}}
elif r['step']==1:a={'action':'trace','input_ids':[1,2,3]}
else:a={'action':'finish'}
print(json.dumps(a))
""")
    dev = [{"input_ids": [1, 2, 3], "target": 4}]
    heldout = [{"input_ids": [2, 3, 4], "target": 5}]
    command = [sys.executable, str(worker)]
    report = run_session(
        model,
        command,
        "test",
        dev,
        heldout,
        mode="program",
        max_forwards=2,
        max_steps=5,
    )
    assert report["budget"]["selection_forwards"] == 2
    assert report["history"][-1]["budget_exhausted"]
    assert report["heldout_sent_to_policy"] is False
    rejected = run_session(
        model, command, "test", dev, mode="direct", max_forwards=3, max_steps=1
    )
    assert "violates direct" in rejected["history"][0]["error"]
    assert rejected["budget"]["selection_forwards"] == 1


def test_failed_policy_command_is_recorded_without_inventing_a_response(
    model, tmp_path
):
    worker = tmp_path / "bad_policy.py"
    worker.write_text('print("not JSON")')
    report = run_session(
        model,
        [sys.executable, str(worker)],
        "test",
        [{"input_ids": [1, 2], "target": 3}],
        max_steps=1,
        max_forwards=3,
    )
    assert report["history"][0]["response"] is None
    assert report["invalid_steps"] == 1
    assert report["policy_usage"] == []
    assert report["audit"]["changed_scalars"] == 0


def test_nonfinite_predictions_cannot_score_as_success(model, monkeypatch):
    from types import SimpleNamespace
    from weightscript.evaluate import score

    monkeypatch.setattr(
        model,
        "forward",
        lambda *args, **kwargs: SimpleNamespace(
            logits=torch.full((1, 2, 13), float("nan"))
        ),
    )
    with pytest.raises(ValueError, match="nonfinite"):
        score(model, [{"input_ids": [1, 2], "target": 0}])
