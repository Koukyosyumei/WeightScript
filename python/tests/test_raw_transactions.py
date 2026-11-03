import pytest
import torch
from test_adapter_capabilities import make_model
from weightscript.core import audit, compile_edits, extract, fingerprint
from weightscript.raw import transaction, transaction_plan


def test_atomic_tied_sparse_and_plan():
    model = make_model("qwen2")
    before = fingerprint(model)
    edit = {
        "op": "scale",
        "tensor": "lm_head.weight",
        "select": [[1, 3], None],
        "value": 0.0,
    }
    plan = transaction_plan(model, [edit])
    assert 0 < plan["sparse_copy_bytes"] < plan["total_storage_bytes"]
    result = transaction(model, [edit], share_untouched=True)
    assert fingerprint(model) == before
    assert (
        result.lm_head.weight.data_ptr() == result.model.embed_tokens.weight.data_ptr()
    )
    assert result.lm_head.weight.data_ptr() != model.lm_head.weight.data_ptr()
    assert (
        result.model.layers[0].mlp.up_proj.weight.data_ptr()
        == model.model.layers[0].mlp.up_proj.weight.data_ptr()
    )
    assert torch.equal(result.lm_head.weight[0], model.lm_head.weight[0])
    assert not result.lm_head.weight[[1, 3]].count_nonzero()
    assert audit(model, result)["changed_scalars"] == 64
    # All operations on one tied storage accumulate without writing source aliases.
    add = dict(edit, op="add", tensor="model.embed_tokens.weight", value=1.0)
    accumulated = transaction(model, [edit, add], share_untouched=True)
    assert torch.all(accumulated.lm_head.weight[[1, 3]] == 1)
    with pytest.raises(ValueError, match="nonfinite"):
        transaction(model, [edit, dict(add, value=float("inf"))], share_untouched=True)
    assert fingerprint(model) == before
    isolated = transaction(model, [edit])
    assert (
        isolated.model.layers[0].mlp.up_proj.weight.data_ptr()
        != model.model.layers[0].mlp.up_proj.weight.data_ptr()
    )


def test_sparse_compiler_unaffected_and_aliases():
    model = make_model("qwen2")
    before = fingerprint(model)
    p = extract(model)
    p["edits"] = [{"op": "attention_scale", "layer": 0, "head": 1, "factor": 0.0}]
    out = compile_edits(model, p, share_untouched=True)
    report = audit(model, out)
    assert {t["tensor"] for t in report["tensors"]} == {
        "model.layers.0.self_attn.q_proj.weight"
    }
    assert (
        out.model.layers[0].self_attn.q_proj.bias.data_ptr()
        != model.model.layers[0].self_attn.q_proj.bias.data_ptr()
    )
    assert out.lm_head.weight.data_ptr() == out.model.embed_tokens.weight.data_ptr()
    assert fingerprint(model) == before


def test_low_rank_dtype_and_buffer_domains():
    model = make_model("llama", torch.bfloat16)
    before = fingerprint(model)
    edit = {
        "op": "low_rank_add",
        "tensor": "lm_head.weight",
        "left": [[1.0] for _ in range(37)],
        "right": [[0.01] * 32],
        "factor": 0.5,
    }
    out = transaction(model, [edit], share_untouched=True)
    assert out.lm_head.weight.dtype == torch.bfloat16
    assert fingerprint(model) == before
    assert audit(model, out)["changed_scalars"] > 0
    model.register_buffer("integer_counter", torch.tensor([1]))
    with pytest.raises(ValueError, match="floating"):
        transaction(
            model,
            [{"op": "add", "tensor": "integer_counter", "select": [None], "value": 1}],
        )
    with pytest.raises(ValueError, match="slice"):
        transaction(
            model,
            [
                {
                    "op": "scale",
                    "tensor": "lm_head.weight",
                    "select": [{"start": -1}, None],
                    "value": 1,
                }
            ],
        )


def test_buffers_isolated_and_view_alias_rejected():
    model = make_model("llama")
    model.register_buffer("counter", torch.tensor([1.0]))
    model.register_buffer("temporary_cache", torch.ones(3), persistent=False)
    edit = {"op": "add", "tensor": "counter", "select": [None], "value": 2.0}
    out = transaction(model, [edit], share_untouched=True)
    assert model.counter.item() == 1
    assert out.counter.item() == 3
    assert out.temporary_cache.data_ptr() != model.temporary_cache.data_ptr()
    parameter = model.lm_head.weight
    model.register_parameter("view_alias", torch.nn.Parameter(parameter[:, :2]))
    with pytest.raises(ValueError, match="overlapping storage views"):
        transaction(model, [edit], share_untouched=True)


def test_overflow_rolls_back_all_edits():
    model = make_model("llama", torch.float16)
    before = fingerprint(model)
    edit = {
        "op": "set",
        "tensor": "lm_head.weight",
        "select": [[1], None],
        "value": 60000.0,
    }
    with pytest.raises(ValueError, match="nonfinite"):
        transaction(
            model, [edit, dict(edit, op="scale", value=60000.0)], share_untouched=True
        )
    assert fingerprint(model) == before


def test_gpt2_sparse_compiler_matches_isolated():
    from transformers import GPT2Config, GPT2LMHeadModel

    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=23,
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_positions=16,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            _attn_implementation="eager",
        )
    ).eval()
    p = extract(model)
    p["edits"] = [
        {"op": "attention_scale", "layer": 0, "head": 1, "factor": 0.5},
        {"op": "head_scale", "layer": 0, "head": 0, "factor": 0.3},
        {"op": "neuron_scale", "layer": 0, "neurons": [2], "factor": -0.2},
    ]
    sparse = compile_edits(model, p, share_untouched=True)
    isolated = compile_edits(model, p)
    assert fingerprint(sparse) == fingerprint(isolated)
    assert (
        sparse.transformer.wte.weight.data_ptr()
        == model.transformer.wte.weight.data_ptr()
    )
    assert sparse.lm_head.weight.data_ptr() == sparse.transformer.wte.weight.data_ptr()
    assert audit(sparse, isolated)["changed_scalars"] == 0
