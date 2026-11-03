import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
from weightscript.core import (
    extract,
    program_forward,
    compile_edits,
    audit,
    fingerprint,
)
from weightscript.evaluate import equivalence


@pytest.fixture(params=["llama", "qwen2"])
def decoder(request):
    torch.manual_seed(130)
    torch.set_num_threads(1)
    config_type, model_type = (
        (LlamaConfig, LlamaForCausalLM)
        if request.param == "llama"
        else (Qwen2Config, Qwen2ForCausalLM)
    )
    c = config_type(
        vocab_size=31,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        _attn_implementation="eager",
        tie_word_embeddings=True,
    )
    return model_type(c).eval()


def test_decoder_native_and_compiled(decoder, tmp_path):
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    p = extract(decoder)
    with torch.no_grad():
        torch.testing.assert_close(
            program_forward(decoder, p, ids),
            decoder(ids, use_cache=False).logits,
            atol=1e-6,
            rtol=1e-5,
        )
    p["edits"] = [
        {"op": "head_scale", "layer": 0, "head": 3, "factor": 0.3},
        {"op": "attention_scale", "layer": 1, "head": 2, "factor": 1.2},
        {"op": "neuron_scale", "layer": 0, "neurons": [1, 3], "factor": -0.5},
        {
            "op": "remove_direction",
            "layer": 1,
            "neurons": [1, 5],
            "direction": [1.0] * 32,
            "strength": 0.8,
        },
    ]
    assert equivalence(
        decoder, p, [{"input_ids": [1, 2, 3, 4]}, {"input_ids": [5, 6, 7]}]
    )["passed"]
    edited = compile_edits(decoder, p)
    assert audit(decoder, edited)["architecture_preserved"]
    edited.save_pretrained(tmp_path)
    reloaded = (
        type(decoder).from_pretrained(tmp_path, attn_implementation="eager").eval()
    )
    assert fingerprint(edited) == fingerprint(reloaded)
    assert audit(edited, reloaded)["changed_scalars"] == 0
    assert (
        reloaded.lm_head.weight.data_ptr()
        == reloaded.model.embed_tokens.weight.data_ptr()
    )
