import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from weightscript.core import extract, program_forward, compile_edits, audit
from weightscript.evaluate import equivalence, benchmark


@pytest.fixture
def model():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=17,
            n_positions=24,
            n_embd=16,
            n_layer=2,
            n_head=2,
            attn_pdrop=0,
            resid_pdrop=0,
            embd_pdrop=0,
            scale_attn_by_inverse_layer_idx=True,
            _attn_implementation="eager",
        )
    ).eval()


def test_original_computation_matches_native_with_padding(model):
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    with torch.no_grad():
        native = model(ids, attention_mask=mask, use_cache=False).logits
        extracted = program_forward(model, extract(model), ids, attention_mask=mask)
    torch.testing.assert_close(native, extracted, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    "edits",
    [
        [{"op": "head_scale", "layer": 0, "head": 1, "factor": 0.0}],
        [{"op": "attention_scale", "layer": 1, "head": 0, "factor": 0.7}],
        [{"op": "neuron_scale", "layer": 0, "neurons": [2, 5, 9], "factor": -0.5}],
        [
            {
                "op": "remove_direction",
                "layer": 1,
                "neurons": [1, 2, 3],
                "direction": [1.0, 2.0] + [0.0] * 14,
                "strength": 0.8,
            }
        ],
        [
            {"op": "neuron_scale", "layer": 0, "neurons": [1, 2], "factor": 0.3},
            {
                "op": "remove_direction",
                "layer": 0,
                "neurons": [1, 2],
                "direction": [1.0] * 16,
                "strength": 1.0,
            },
            {"op": "head_scale", "layer": 1, "head": 0, "factor": 0.5},
        ],
    ],
)
def test_compiled_edits_match_program_and_preserve_other_tensors(model, edits):
    p = extract(model)
    p["edits"] = edits
    rows = [{"input_ids": [1, 2, 3, 4]}, {"input_ids": [2, 4, 5]}]
    assert equivalence(model, p, rows)["passed"]
    updated = compile_edits(model, p)
    assert audit(model, updated)["architecture_preserved"]
    # Source checkpoint stays bitwise unchanged, tied embeddings stay tied.
    assert extract(model)["base_sha256"] == p["base_sha256"]
    assert (
        updated.lm_head.weight.data_ptr() == updated.transformer.wte.weight.data_ptr()
    )
    edited_layers = {e["layer"] for e in edits}
    for name, value in model.state_dict().items():
        if not any(name.startswith(f"transformer.h.{i}.") for i in edited_layers):
            assert torch.equal(value, updated.state_dict()[name])


def test_checkpoint_reload(model, tmp_path):
    p = extract(model)
    p["edits"] = [{"op": "head_scale", "layer": 0, "head": 0, "factor": 0}]
    updated = compile_edits(model, p)
    updated.save_pretrained(tmp_path)
    reloaded = GPT2LMHeadModel.from_pretrained(
        tmp_path, attn_implementation="eager"
    ).eval()
    ids = torch.tensor([[1, 3, 4]])
    with torch.no_grad():
        torch.testing.assert_close(
            updated(ids).logits, reloaded(ids).logits, atol=0, rtol=0
        )


def test_reject_stale_or_unsupported_edits(model):
    p = extract(model)
    p["base_sha256"] = "wrong"
    with pytest.raises(ValueError):
        compile_edits(model, p)
    for edit in [
        {"op": "head_scale", "layer": 0, "head": 99, "factor": 0},
        {"op": "neuron_scale", "layer": 0, "neurons": [-1], "factor": 0},
        {"op": "neuron_scale", "layer": 0, "neurons": [1, 1], "factor": 0},
        {"op": "head_scale", "layer": 0, "head": 0, "factor": float("nan")},
        {
            "op": "remove_direction",
            "layer": 0,
            "neurons": [1],
            "direction": [0] * 16,
            "strength": 1,
        },
    ]:
        p = extract(model)
        p["edits"] = [edit]
        with pytest.raises(ValueError):
            compile_edits(model, p)


def test_benchmark_budget_and_split(model):
    dev = [{"input_ids": [1, 2], "target": 3}, {"input_ids": [4, 5], "target": 6}]
    test = [{"input_ids": [2, 3], "target": 4}]
    edits = [{"op": "head_scale", "layer": 0, "head": 0, "factor": 0}]
    report = benchmark(
        model,
        {"direct": [[], edits], "program": [[], edits]},
        dev,
        test,
        batch_size=1,
        budget=2,
    )
    for method in report["methods"]:
        assert method["development_forward_calls"] == 2
        assert len(method["trials"]) == 1
    with pytest.raises(ValueError):
        benchmark(model, {"x": [[]]}, dev, dev)


def test_undo_survives_reload_and_rejects_wrong_checkpoint(model, tmp_path):
    from weightscript.checkpoint import save_undo, undo
    from weightscript.core import fingerprint

    p = extract(model)
    p["edits"] = [{"op": "neuron_scale", "layer": 0, "neurons": [3, 7], "factor": 0.0}]
    updated = compile_edits(model, p)
    updated.save_pretrained(tmp_path)
    save_undo(model, updated, tmp_path)
    reloaded = GPT2LMHeadModel.from_pretrained(
        tmp_path, attn_implementation="eager"
    ).eval()
    assert fingerprint(updated) == fingerprint(reloaded)
    restored = undo(reloaded, tmp_path)
    assert fingerprint(restored) == fingerprint(model)
    with pytest.raises(ValueError):
        undo(model, tmp_path)


def test_direct_tensor_baseline_matches_program_and_reports_different_proposal_surface(
    model,
):
    from weightscript.core import apply_raw_edits

    rows = [0, 1, 2, 3, 4, 5, 6, 7]
    raw = [
        {
            "op": "scale",
            "tensor": "transformer.h.0.attn.c_proj.weight",
            "rows": rows,
            "factor": 0.5,
        }
    ]
    p = extract(model)
    p["edits"] = [{"op": "head_scale", "layer": 0, "head": 0, "factor": 0.5}]
    direct = apply_raw_edits(model, raw)
    compiled = compile_edits(model, p)
    assert audit(direct, compiled)["changed_scalars"] == 0


def test_audit_counts_tied_storage_once_and_checks_architecture(model):
    from weightscript.core import apply_raw_edits

    raw = [
        {"op": "scale", "tensor": "transformer.wte.weight", "rows": [0], "factor": 0}
    ]
    changed = apply_raw_edits(model, raw)
    report = audit(model, changed)
    assert report["changed_scalars"] == 16
    assert report["tensors"][0]["aliases"] == ["lm_head.weight"]
    changed.config.n_head = 1
    with pytest.raises(ValueError):
        audit(model, changed)
