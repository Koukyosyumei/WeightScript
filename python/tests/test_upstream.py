import os
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from weightscript.backends.upstream import decompile, execute_artifact


@pytest.mark.skipif(
    not os.environ.get("WEIGHTSCRIPT_UPSTREAM"),
    reason="pinned upstream checkout not provided",
)
def test_actual_upstream_hooks_reconstruct_multi_layer_model():
    torch.manual_seed(5)
    torch.set_num_threads(1)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=9,
            n_positions=16,
            n_embd=16,
            n_layer=2,
            n_head=2,
            attn_pdrop=0,
            resid_pdrop=0,
            embd_pdrop=0,
            _attn_implementation="eager",
        )
    ).eval()
    rows = [{"input_ids": [1, 2, 3]}, {"input_ids": [2, 3, 4]}]
    result = decompile(
        model, os.environ["WEIGHTSCRIPT_UPSTREAM"], rows, max_trials=2, min_fidelity=1.0
    )
    assert result.fidelity["initial"]["max_logit_error"] < 1e-5
    assert result.provenance["revision"] == "787398b0f4393a90ad02cd8e71be48d1e85d2b21"
    assert "Attention" in result.nodes[-1].attributes["source"]

    before = model(
        torch.tensor([row["input_ids"] for row in rows]), use_cache=False
    ).logits
    replay = execute_artifact(
        model,
        result,
        os.environ["WEIGHTSCRIPT_UPSTREAM"],
        torch.tensor([row["input_ids"] for row in rows]),
    )
    assert torch.equal(before[:, -1].argmax(-1), replay[:, -1].argmax(-1))
