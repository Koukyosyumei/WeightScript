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


@pytest.mark.skipif(
    not os.environ.get("WEIGHTSCRIPT_UPSTREAM"), reason="pinned checkout required"
)
def test_three_real_engines_json_replay_and_opaque_paths():
    import json

    from weightscript.backends.upstream_stages import decompile_stages, execute_stages
    from weightscript.core import fingerprint

    torch.manual_seed(5)
    torch.set_num_threads(2)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=8,
            n_positions=8,
            n_embd=8,
            n_layer=2,
            n_head=1,
            _attn_implementation="eager",
        )
    ).eval()
    source = fingerprint(model)
    rows = [{"input_ids": [1, 2, 3]}, {"input_ids": [2, 3, 4]}]
    final = [{"input_ids": [4, 2, 1]}]
    artifact = decompile_stages(
        model,
        os.environ["WEIGHTSCRIPT_UPSTREAM"],
        rows,
        final,
        max_trials=1,
        calibration_steps=1,
        min_fidelity=1,
    )
    serialized = json.loads(json.dumps(artifact.to_dict()))
    assert artifact.provenance["completed_stages"] == ["component", "full_path", "qk"]
    assert fingerprint(model) == source
    for stage in ("full_path", "qk"):
        output = execute_stages(
            model,
            serialized,
            os.environ["WEIGHTSCRIPT_UPSTREAM"],
            torch.tensor([[1, 2, 3]]),
            stage,
        )
        assert output.shape == (1, 3, 8)
        assert torch.isfinite(output).all()
    stages = serialized["nodes"][-1]["attributes"]["stages"]
    assert stages[-1]["calibration"]["examples"] == 2
    assert "counterexamples" in stages[-1]["evaluation"]
    assert "primitive_matches" in serialized["nodes"][-1]["attributes"]
    stages[-1]["masks"][0]["value"] = 2
    with pytest.raises(ValueError, match="masks"):
        execute_stages(
            model,
            serialized,
            os.environ["WEIGHTSCRIPT_UPSTREAM"],
            torch.tensor([[1, 2, 3]]),
        )


def test_stage_summary_reads_measured_fields():
    from weightscript.backends.pipeline import summarize_stages

    artifact = {
        "backend": "decompiling_transformers.three_stage",
        "provenance": {
            "checkpoint_sha256": "abc",
            "completed_stages": ["component", "full_path", "qk"],
            "compatibility_adaptations": ["example"],
        },
        "fidelity": {
            "accepted": False,
            "claim": "measured",
            "normalization": {"positions": "0..n-1"},
        },
        "nodes": [
            {"attributes": {}},
            {
                "attributes": {
                    "stages": [
                        {
                            "name": "component",
                            "accepted": True,
                            "artifact": {
                                "fidelity": {
                                    "component_edges_before": 2,
                                    "component_edges_after": 1,
                                    "trials": [{"edge": []}],
                                    "calibration": {
                                        "final_token_agreement": 1.0,
                                        "max_logit_error": 3.0,
                                    },
                                    "evaluation": {"final_token_agreement": 0.5},
                                    "normalization_assumption_passed": True,
                                    "logit_reconstruction": False,
                                }
                            },
                        },
                        {
                            "name": "full_path",
                            "accepted": False,
                            "edges_before": 2,
                            "edges_after": 2,
                            "trials": [{}],
                            "calibration": {
                                "final_token_agreement": 1.0,
                                "max_logit_error": 4.0,
                            },
                            "evaluation": {"final_token_agreement": 0.25},
                            "renderer_adaptation": None,
                        },
                    ],
                    "primitive_matches": [
                        {
                            "id": "uniform.0.0",
                            "interpretation": "primitive",
                            "attributes": {},
                        },
                        {
                            "id": "qk.0.0.0",
                            "interpretation": "opaque",
                            "attributes": {
                                "claim": "left opaque",
                                "best_match": {
                                    "primitive": "diag",
                                    "relative_error": 0.4,
                                    "operation": "k==q",
                                },
                            },
                        },
                    ],
                }
            },
        ],
    }
    summary = summarize_stages(artifact)
    assert summary["accepted"] is False
    assert summary["stages"][0]["search_complete"] is False
    assert summary["stages"][1]["evaluation_agreement"] == 0.25
    assert summary["primitive_interpretations"] == {"primitive": 1, "opaque": 1}
    assert summary["primitive_rejections"][0]["relative_error"] == 0.4
    assert summary["stages"][0]["logit_reconstruction"] is False


def test_stage_public_configuration_rejected_before_source_access():
    from weightscript.backends.upstream_stages import decompile_stages

    with pytest.raises(ValueError, match="budget"):
        decompile_stages(None, "unused", [], max_trials=-1)
    with pytest.raises(ValueError, match="budget"):
        decompile_stages(None, "unused", [], calibration_steps=10001)


def test_answer_span_acceptance_rejects_first_token_and_failed_layernorm():
    from weightscript.backends.upstream import answer_logit_positions, stage_accepted

    assert answer_logit_positions(6, 3) == [2, 3, 4]
    first_token_only = stage_accepted(
        {"answer_position_agreement": 1.0},
        {"answer_position_agreement": 1.0},
        {"answer_position_agreement": 21 / 144},
        0.99,
    )
    assert first_token_only["accepted"] is False
    failed_norm = stage_accepted(
        {"answer_position_agreement": 22 / 24},
        {"answer_position_agreement": 1.0},
        {"answer_position_agreement": 1.0},
        0.99,
    )
    assert failed_norm["normalization_assumption_passed"] is False
    assert failed_norm["accepted"] is False


def test_retained_greedy_failures_remain_in_the_result_files():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "docs" / "results" / "decompilation"
    copy = json.loads((root / "copy-summary.json").read_text())
    assert copy["replay"]["full_path"]["final"]["stage_vs_native"] < 0.2
    assert copy["replay"]["qk"]["final"]["positions"] == 144
    majority = json.loads((root / "majority-summary.json").read_text())
    assert majority["causal_summary"]["accepted"] is False
    assert majority["replay"]["full_path"]["final"]["stage_vs_native"] < 0.99
    sort = json.loads((root / "sort-summary.json").read_text())
    component = next(
        stage
        for stage in sort["causal_summary"]["stages"]
        if stage["name"] == "component"
    )
    assert component["normalization_assumption_passed"] is False
    assert sort["replay"]["full_path"]["oracle"]["stage_vs_native"] < 0.2


def test_coefficient_budget_is_rejected_before_source_access():
    from weightscript.backends.upstream_search import decompile_coefficient

    with pytest.raises(ValueError, match="budget"):
        decompile_coefficient(None, "unused", [{"input_ids": [1, 2]}], steps=65)


@pytest.mark.skipif(
    not os.environ.get("WEIGHTSCRIPT_UPSTREAM"), reason="pinned checkout required"
)
def test_coefficient_search_meters_steps_and_keeps_failed_mlp_opaque():
    import json

    from weightscript.backends.pipeline import summarize_stages
    from weightscript.backends.upstream_search import (
        decompile_coefficient,
        execute_coefficient,
    )
    from weightscript.core import fingerprint

    torch.manual_seed(5)
    torch.set_num_threads(2)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=8,
            n_positions=8,
            n_embd=8,
            n_layer=2,
            n_head=1,
            _attn_implementation="eager",
        )
    ).eval()
    source = fingerprint(model)
    rows = [
        {"input_ids": [1, 2, 3, 4], "answer_start": 2},
        {"input_ids": [2, 3, 4, 1], "answer_start": 2},
    ]
    held = [{"input_ids": [4, 1, 2, 3], "answer_start": 2}]
    artifact = decompile_coefficient(
        model,
        os.environ["WEIGHTSCRIPT_UPSTREAM"],
        rows,
        held,
        steps=1,
        pretrain_steps=1,
        batch_size=2,
        max_seconds=120,
        max_mlp_rows=8,
        max_behavior_tests=1,
        max_feature_dim=32,
        min_fidelity=0.99,
    )
    assert fingerprint(model) == source
    serialized = json.loads(json.dumps(artifact.to_dict()))
    assert serialized["provenance"]["limits"]["countdown_extension"] is False
    assert "HTCondor not used" in serialized["provenance"]["driver"]
    stages = serialized["nodes"][-1]["attributes"]["stages"]
    assert [stage["name"] for stage in stages] == ["component", "full_path", "qk"]
    for stage in stages:
        assert stage["meter"]["optimizer_steps"] <= 1
        assert stage["meter"]["pretrain_steps"] <= 1
        assert stage["calibration"]["acceptance_metric"] == "answer_positions"
        if not stage["normalization_assumption_passed"]:
            assert stage["accepted"] is False
        if stage["evaluation"]["answer_position_agreement"] < 0.99:
            assert stage["accepted"] is False
    replacement = stages[-1]["mlp_replacement"]
    for row in replacement["accepted"]:
        assert row["passed"] is True
        assert row["causal_agreement"] >= 0.99
        assert row["heldout_vs_native"] >= 0.99
    summary = summarize_stages(serialized)
    assert summary["stages"][-1]["mlp_installed"] == replacement["installed"]
    replay = execute_coefficient(
        model,
        serialized,
        os.environ["WEIGHTSCRIPT_UPSTREAM"],
        torch.tensor([rows[0]["input_ids"]]),
    )
    assert replay.shape == (1, 4, 8)
    assert torch.isfinite(replay).all()
    assert fingerprint(model) == source
