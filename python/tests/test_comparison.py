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


def test_first_answer_token_is_not_full_credit():
    from weightscript.evaluate import score, score_objective

    class Scripted(torch.nn.Module):
        def forward(self, input_ids, use_cache=False):
            batch, seq = input_ids.shape
            logits = torch.full((batch, seq, 8), -10.0)
            if seq > 3:
                logits[:, 3, 0] = 5
            if seq > 4:
                logits[:, 4, 1] = 5
            return type("Out", (), {"logits": logits})()

    span = [{"input_ids": [2, 0, 1, 3], "targets": [0, 4]}]
    metrics = score_objective(Scripted(), span, objective="sequence")
    assert metrics["labeled"] == 2
    assert metrics["accuracy"] == 0.5
    assert metrics["full_span"] is True
    prompt = [{"input_ids": [2, 0, 1, 3], "target": 0}]
    assert score(Scripted(), prompt)["labeled"] == 1
    assert score(Scripted(), prompt)["accuracy"] == 1


def test_sequence_forward_groups_use_prepared_length():
    from weightscript.evaluate import count_forwards

    rows = [
        {"input_ids": [1, 2, 3], "targets": [4, 5]},
        {"input_ids": [1, 2, 3], "targets": [4, 6]},
        {"input_ids": [1], "targets": [2, 3, 4]},
    ]
    assert count_forwards(rows, 32, "sequence") == 2
    assert count_forwards([{"input_ids": [1, 2], "target": 3}], 32) == 1


def test_disjoint_seed_windows_do_not_repeat():
    from weightscript.comparison import disjoint_seed_windows

    splits = disjoint_seed_windows(list(range(300)), 16, [0, 1, 2], 4)
    sequences = []
    for seed, rows in splits.items():
        assert rows[0]["window_index"] == seed * 4
        for row in rows:
            sequences.append(tuple(row["input_ids"] + row["targets"]))
            assert row["task"] == "language" and row["answer_length"] == 1
    assert len(sequences) == len(set(sequences)) == 12
    with pytest.raises(ValueError, match="windows"):
        disjoint_seed_windows(list(range(20)), 4, [0, 1], 3)


def test_compare_scores_every_target_and_rejects_a_short_span(model, monkeypatch):
    from weightscript.comparison import require_complete_spans

    def make(marker, count):
        return [
            {"input_ids": [1, marker, i], "targets": [4, 5], "answer_length": 2}
            for i in range(count)
        ]

    report = compare(model, make(2, 3), make(3, 3), make(4, 2), seed=1, trials=1)
    assert "every target token" in report["grader"]
    assert report["min_answer_tokens"] == 2
    assert report["labeled_positions"] == 16
    untouched = next(row for row in report["methods"] if row["method"] == "untouched")
    assert untouched["development"]["labeled"] == 6
    assert untouched["heldout"]["labeled"] == 6
    assert untouched["heldout"]["min_answer_tokens"] == 2
    copied = task_rows("unique_copy", 0, [4], 2)
    require_complete_spans(copied)
    assert all(len(row["targets"]) == row["answer_length"] == 5 for row in copied)
    require_complete_spans(task_rows("bin_majority", 0, [3], 2))
    short = {
        "input_ids": copied[0]["input_ids"],
        "targets": copied[0]["targets"][:1],
        "task": "unique_copy",
        "answer_length": 1,
    }
    called = []
    monkeypatch.setattr(
        "weightscript.comparison.score_answers",
        lambda *args, **kwargs: called.append(1),
    )
    with pytest.raises(ValueError, match="truncated"):
        compare(
            model,
            [short],
            [copied[1]],
            task_rows("unique_copy", 3, [3], 2),
            seed=0,
            trials=1,
        )
    assert called == []


def test_preserved_passage_hash_and_disjoint_chapters():
    import hashlib
    import json
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "data"
        / "public_domain_lm.json"
    )
    if not path.is_file():
        pytest.skip("passage data is repository data")
    passage = json.loads(path.read_text())
    legacy = hashlib.sha256(
        (passage["target_text"] + "\n" + passage["collateral_text"]).encode()
    ).hexdigest()
    assert (
        legacy
        == passage["sha256"]
        == "59b9b35994bdc1df809fdf02be72d208ba7fe0f5c865089f5a78031f0ef45d89"
    )
    block = passage["disjoint_evaluation"]
    body = "\n".join(
        [block["development_text"], block["heldout_text"], block["collateral_text"]]
    )
    assert hashlib.sha256(body.encode()).hexdigest() == block["sha256"]
    assert block["development_chapter"] == "I"
    assert block["heldout_chapter"] == "III"
    assert block["collateral_chapter"] == "X"
    assert block["development_text"].startswith(
        "It is a truth universally acknowledged"
    )
    assert (
        len(
            set(
                block[name]
                for name in (
                    "development_text",
                    "heldout_text",
                    "collateral_text",
                )
            )
        )
        == 3
    )


def test_backend_contracts_fail_closed():
    from weightscript.adapters import CAPABILITIES

    assert set(CAPABILITIES) == set(CONTRACTS) - {"gpt2"}
    for name, capability in CAPABILITIES.items():
        assert set(backend_contract(name)["rope"]) == set(capability["rope"])
    with pytest.raises(ValueError, match="unsupported"):
        backend_contract("qwen3_5")
