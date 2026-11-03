import pytest
import torch

pytest.importorskip("z3")
from transformers import GPT2Config, GPT2LMHeadModel
from weightscript.synthesis import compile_rule, rule_predict
from weightscript.symbolic import synthesize_binary
from weightscript.generation import evaluate_generation
from weightscript.core import fingerprint
from weightscript.explain import explain


def test_rule_validation_and_unsat_do_not_mutate_source():
    torch.manual_seed(8)
    torch.set_num_threads(1)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=6,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            _attn_implementation="eager",
        )
    ).eval()
    source = fingerprint(model)
    with pytest.raises(ValueError):
        rule_predict({"op": "other"}, [0, 1])
    with pytest.raises(ValueError):
        compile_rule(
            model,
            {"op": "count_threshold", "threshold": 0.5, "invert": True},
            [{"input_ids": [2, 0, 3]}],
            verification=[{"input_ids": [2, 0, 3]}],
        )
    assert fingerprint(model) == source
    report = evaluate_generation(model, [{"input_ids": [1, 2]}], max_new_tokens=2)
    assert report["forward_calls"] <= 2
    assert report["results"][0]["tokens"]
    assert explain(model, model).nodes[-1].op == "patched_checkpoint"


def test_symbolic_synthesis_uses_opaque_fallback():
    torch.manual_seed(3)
    torch.set_num_threads(1)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=6,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            _attn_implementation="eager",
        )
    ).eval()
    rows = [{"input_ids": [2, 0, 0, 3]}, {"input_ids": [2, 1, 1, 3]}]
    artifact = synthesize_binary(model, rows, min_fidelity=1.0)
    artifact.validate()
    assert "behavioral" in artifact.provenance["method"]
