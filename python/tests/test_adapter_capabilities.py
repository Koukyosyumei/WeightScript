"""Generated checkpoints compare independent execution to native Transformers."""

import pytest
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from weightscript.adapters import rotary, validate
from weightscript.core import (
    audit,
    compile_edits,
    extract,
    fingerprint,
    program_forward,
)

FAMILIES = {
    "llama": (LlamaConfig, LlamaForCausalLM),
    "qwen2": (Qwen2Config, Qwen2ForCausalLM),
    "mistral": (MistralConfig, MistralForCausalLM),
}


def make_model(family, dtype=torch.float32, **kwargs):
    torch.manual_seed(51)
    torch.set_num_threads(2)
    config, model = FAMILIES[family]
    options = {
        "vocab_size": 37,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 8,
        "attention_dropout": 0.0,
        "tie_word_embeddings": True,
        "_attn_implementation": "eager",
    }
    if family == "mistral":
        options["head_dim"] = 8
    options.update(kwargs)
    return model(config(**options)).to(dtype=dtype).eval()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "rope",
    [
        None,
        {"rope_type": "linear", "factor": 2.0},
        {"rope_type": "dynamic", "factor": 2.0},
        {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 4,
        },
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_native_rotary_gqa_dtype(family, rope, dtype):
    model = make_model(family, dtype, rope_scaling=rope)
    p = extract(model)
    ids = torch.arange(24).reshape(2, 12) % 37  # dynamic threshold crossed
    mask = torch.ones_like(ids)
    mask[0, -3:] = 0
    mask[1, :2] = 0
    with torch.no_grad():
        native = model(ids, attention_mask=mask, use_cache=False).logits
    actual = program_forward(model, p, ids, mask)
    tolerance = {torch.float32: 2e-6, torch.float16: 8e-4, torch.bfloat16: 6e-3}[dtype]
    torch.testing.assert_close(actual, native, atol=tolerance, rtol=tolerance)
    # Compare rotary tensors directly so tiny random logits cannot hide a wrong scale.
    with torch.no_grad():
        native_cos, native_sin = model.model.rotary_emb(
            torch.empty(1, 12, 32, dtype=dtype), torch.arange(12)[None]
        )
    cos, sin = rotary(model.config, 12, 8, torch.device("cpu"), dtype)
    torch.testing.assert_close(cos[0, 0], native_cos[0], atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(sin[0, 0], native_sin[0], atol=tolerance, rtol=tolerance)
    # Native dynamic rotary buffers can grow/shrink, but persistent identity must not.
    assert fingerprint(model) == p["base_sha256"]
    p["edits"] = [
        {"op": "attention_scale", "layer": 0, "head": 3, "factor": 0.5},
        {"op": "head_scale", "layer": 1, "head": 1, "factor": 0.7},
        {"op": "neuron_scale", "layer": 0, "neurons": [2, 7], "factor": -0.5},
    ]
    compiled = compile_edits(model, p)
    with torch.no_grad():
        expected = compiled(ids, attention_mask=mask, use_cache=False).logits
    torch.testing.assert_close(
        program_forward(model, p, ids, mask), expected, atol=tolerance, rtol=tolerance
    )
    assert audit(model, compiled)["architecture_preserved"]


@pytest.mark.parametrize("family", ["qwen2", "mistral"])
def test_windows_and_mixed_layers(family):
    kwargs = {"sliding_window": 3}
    if family == "qwen2":
        kwargs.update(
            use_sliding_window=True, layer_types=["full_attention", "sliding_attention"]
        )
    model = make_model(family, **kwargs)
    ids = torch.arange(7)[None]
    p = extract(model)
    actual, traces = program_forward(model, p, ids, capture=True)
    with torch.no_grad():
        native = model(ids, use_cache=False).logits
    torch.testing.assert_close(actual, native, atol=2e-6, rtol=2e-5)
    assert torch.count_nonzero(traces[1]["attention"][..., -1, :-3]) == 0
    if family == "qwen2":
        assert torch.count_nonzero(traces[0]["attention"][..., -1, :-3]) > 0


def test_domains_and_explicit_capabilities():
    model = make_model("llama")
    p = extract(model)
    for ids in [
        torch.ones(1, 2),
        torch.tensor([[-1]]),
        torch.tensor([[37]]),
        torch.empty(1, 0, dtype=torch.long),
    ]:
        with pytest.raises(ValueError, match="token domain"):
            program_forward(model, p, ids)
    with pytest.raises(ValueError, match="mask"):
        program_forward(
            model, p, torch.ones(1, 2, dtype=torch.long), torch.tensor([[1, 2]])
        )
    model.config.rope_scaling = {"rope_type": "yarn", "factor": 2.0}
    with pytest.raises(ValueError, match="unsupported RoPE"):
        validate(model)
    model.config.model_type = "qwen3_5"
    with pytest.raises(ValueError, match="no semantic adapter"):
        validate(model)
    assert "qwen3_5" not in CONFIG_MAPPING_NAMES  # pinned 4.57.1 feasibility evidence


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_cuda_native():
    model = make_model("qwen2", torch.float16).cuda()
    ids = torch.tensor([[1, 2, 3]], device="cuda")
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
    torch.testing.assert_close(
        program_forward(model, extract(model), ids), expected, atol=8e-4, rtol=8e-4
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_lower_precision_checkpoint_roundtrip(family, tmp_path):
    from weightscript.core import load_model

    model = make_model(family, torch.bfloat16)
    p = extract(model)
    p["edits"] = [
        {
            "op": "remove_direction",
            "layer": 0,
            "neurons": [1, 4],
            "direction": [1e-20] * 32,
            "strength": 0.5,
        }
    ]
    compiled = compile_edits(model, p, share_untouched=True)
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        native = compiled(ids, use_cache=False).logits
    torch.testing.assert_close(
        program_forward(model, p, ids), native, atol=6e-3, rtol=6e-3
    )
    compiled.save_pretrained(tmp_path)
    reloaded = load_model(tmp_path, dtype=torch.bfloat16)
    assert fingerprint(compiled) == fingerprint(reloaded)
    assert audit(compiled, reloaded)["changed_scalars"] == 0
    assert (
        reloaded.lm_head.weight.data_ptr()
        == reloaded.model.embed_tokens.weight.data_ptr()
    )


def test_invalid_variants_rejected():
    model = make_model("llama")
    model.config.sliding_window = 3
    with pytest.raises(ValueError, match="does not implement sliding"):
        extract(model)
    model.config.sliding_window = None
    for scaling in [
        {"rope_type": "linear", "factor": 0.0},
        {"rope_type": "dynamic", "factor": float("nan")},
        {"rope_type": "llama3", "factor": 8.0},
    ]:
        model.config.rope_scaling = scaling
        with pytest.raises(ValueError, match="RoPE"):
            extract(model)
    model.config.rope_scaling = None
    model.model.layers[0].self_attn.q_proj.to(torch.float16)
    with pytest.raises(ValueError, match="homogeneous"):
        extract(model)


@pytest.mark.parametrize("family", ["llama", "qwen2"])
def test_dynamic_expansion_and_reset(family):
    model = make_model(family, rope_scaling={"rope_type": "dynamic", "factor": 4.0})
    p = extract(model)
    for length in [16, 4, 20]:
        ids = torch.arange(length)[None] % 37
        with torch.no_grad():
            expected = model(ids, use_cache=False).logits
        torch.testing.assert_close(
            program_forward(model, p, ids), expected, atol=2e-6, rtol=2e-5
        )


def test_mistral_default_head_dimension():
    model = make_model("mistral", head_dim=None)
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        native = model(ids, use_cache=False).logits
    torch.testing.assert_close(
        program_forward(model, extract(model), ids), native, atol=2e-6, rtol=2e-5
    )
