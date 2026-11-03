import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from weightscript.comparison import compare, proposals_for, task_rows
from weightscript.contracts import CONTRACTS, backend_contract
from weightscript.evaluate import score_answers


@pytest.fixture
def model():
    torch.manual_seed(3)
    torch.set_num_threads(1)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=16,
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            n_inner=16,
            attn_pdrop=0,
            resid_pdrop=0,
            embd_pdrop=0,
            _attn_implementation="eager",
        )
    ).eval()


def test_token_windows_are_disjoint():
    from weightscript.comparison import token_windows

    rows = token_windows(list(range(40)), prompt_length=4, count=4)
    ids = [tuple(row["input_ids"] + row["targets"]) for row in rows]
    assert len(set(ids)) == 4
    assert rows[0]["targets"] == [4]
    with pytest.raises(ValueError):
        token_windows([1, 2, 3], 4, 1)


def test_answer_score_and_reference_disagreement(model):
    rows = [{"input_ids": [2, 0, 1, 3], "targets": [1, 4]}]
    alone = score_answers(model, rows)
    same = score_answers(model, rows, reference=model)
    assert 0 <= alone["accuracy"] <= 1
    assert same["token_disagreement"] == 0
    assert same["reference_forward_calls"] == same["forward_calls"]


def test_comparison_keeps_heldout_out_of_selection(model, monkeypatch):
    dev = task_rows("bin_majority", 1, [3], 4)
    heldout = task_rows("bin_majority", 2, [4], 4)
    collateral = task_rows("bin_majority", 3, [5], 3)
    held_ids = {tuple(row["input_ids"]) for row in heldout}
    calls = []
    original = score_answers

    def spy(scored, rows, batch_size=32, reference=None):
        calls.append(
            (tuple(tuple(row["input_ids"]) for row in rows), reference is not None)
        )
        return original(scored, rows, batch_size, reference)

    monkeypatch.setattr("weightscript.comparison.score_answers", spy)
    report = compare(model, dev, heldout, collateral, seed=7, trials=1)
    selected = [call for call in calls if call[1] is False]
    assert selected and all(not (set(call[0]) & held_ids) for call in selected)
    assert any(call[1] for call in calls)
    by_name = {row["method"]: row for row in report["methods"]}
    assert by_name["direct"]["changed_scalars"] == by_name["program"]["changed_scalars"]
    assert by_name["untouched"]["changed_scalars"] == 0
    assert "not evidence of superiority" in report["claim"]
    with pytest.raises(ValueError, match="overlap"):
        compare(model, dev, dev, collateral, seed=1, trials=1)
    with pytest.raises(ValueError, match="unsupported"):
        proposals_for(model, "invented", seed=1)


def test_backend_contracts_fail_closed():
    from weightscript.adapters import CAPABILITIES

    assert set(CAPABILITIES) == set(CONTRACTS) - {"gpt2"}
    for name, capability in CAPABILITIES.items():
        assert set(backend_contract(name)["rope"]) == set(capability["rope"])
    with pytest.raises(ValueError, match="unsupported"):
        backend_contract("qwen3_5")
