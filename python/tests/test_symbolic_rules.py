"""Public semantic configurations and bounded native lowering checks."""
import copy
import math

import pytest
from weightscript.backends.rules import rule_predict, validate_rule


def test_copy_comparison_count_selection_and_branch():
    ids = [8, 2, 4, 2, 9]
    copy_rule = {"op": "token_at", "index": -1, "default": 0, "ignore_tokens": [8, 9]}
    assert rule_predict(copy_rule, ids) == 2
    assert rule_predict(dict(copy_rule, index=8), ids) == 0
    compare = {"op": "token_compare", "left": 1, "right": 2, "comparison": "lt",
               "if_true": 1, "if_false": 0}
    assert rule_predict(compare, ids) == 1
    count = {"op": "count_token", "token": 2, "threshold": 2, "comparison": "eq",
             "if_true": 1, "if_false": 0}
    assert rule_predict(count, ids) == 1
    select = {"op": "select_token", "token": 2, "comparison": "gt", "which": "first", "default": 0,
              "ignore_tokens": [8, 9]}
    assert rule_predict(select, ids) == 4
    branch = {"op": "branch", "condition": count, "then": copy_rule, "else": select}
    assert rule_predict(branch, ids) == 2
    assert rule_predict(dict(branch, condition=dict(count, threshold=3)), ids) == 4


@pytest.mark.parametrize("rule", [
    {"op": "count_threshold", "threshold": math.nan, "invert": False},
    {"op": "token_at", "index": True, "default": 0},
    {"op": "token_at", "index": 1, "default": -1},
    {"op": "token_at", "index": 1, "default": 0, "arbitrary": 1},
    {"op": "token_at", "index": 1, "default": 0, "ignore_tokens": [True]},
    {"op": "select_token", "token": 1, "comparison": "eval", "which": "first", "default": 0},
])
def test_invalid_public_rules(rule):
    with pytest.raises(ValueError):
        validate_rule(rule)


def test_undefined_comparison_and_depth_bound():
    compare = {"op": "token_compare", "left": 0, "right": 5, "comparison": "eq",
               "if_true": 1, "if_false": 0}
    with pytest.raises(ValueError, match="outside"):
        rule_predict(compare, [1])
    leaf = {"op": "token_at", "index": 0, "default": 0}
    rule = leaf
    for _ in range(18):
        rule = {"op": "branch", "condition": leaf, "then": rule, "else": leaf}
    with pytest.raises(ValueError, match="nested"):
        validate_rule(rule)


def tiny_model():
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(19)
    torch.set_num_threads(2)
    return GPT2LMHeadModel(GPT2Config(
        vocab_size=5, n_positions=12, n_embd=12, n_layer=1, n_head=1,
        attn_pdrop=0, resid_pdrop=0, embd_pdrop=0,
        _attn_implementation="eager")).eval()


def test_multiclass_copy_native_compilation_preserves_source():
    import torch
    from weightscript.core import fingerprint
    from weightscript.synthesis import compile_rule
    model = tiny_model()
    source = fingerprint(model)
    rule = {"op": "token_at", "index": 0, "default": 0}
    rows = [{"input_ids": [t, 4]} for t in (0, 1, 2)]
    edited, report = compile_rule(model, rule, rows)
    assert report["passed"]
    import json

    from weightscript.synthesis import replay_patch
    replayed = replay_patch(model, json.loads(json.dumps(report)))
    assert fingerprint(replayed) == fingerprint(edited)
    damaged = copy.deepcopy(report)
    damaged["coefficients"][0] += 1
    with pytest.raises(ValueError, match="checksum"):
        replay_patch(model, damaged)
    assert report["patch_rank_bound"] > 1
    assert report["solver"]["rational_violation_check"] == "unsat"
    assert fingerprint(model) == source
    assert edited(torch.tensor([r["input_ids"] for r in rows])).logits[:, -1].argmax(-1).tolist() == [0, 1, 2]
    assert {r["tensor"] for r in report["audit"]["tensors"]} <= {
        "transformer.h.0.mlp.c_proj.weight", "transformer.h.0.mlp.c_proj.bias", "transformer.ln_f.bias"}


def test_semantic_matching_evidence_binding_and_opaque_retention():
    from weightscript.core import fingerprint
    from weightscript.symbolic import synthesize_rules
    model = tiny_model()
    rows = [{"input_ids": [1, 2]}, {"input_ids": [2, 3]}]
    candidates = [{"op": "token_at", "index": -1, "default": 0},
                  {"op": "token_at", "index": -2, "default": 0}]
    with pytest.raises(ValueError, match="different checkpoint"):
        synthesize_rules(model, candidates, rows, causal_artifact={})
    with pytest.raises(ValueError, match="overlap"):
        synthesize_rules(model, candidates, rows, rows)
    evidence = {"provenance": {"checkpoint_sha256": fingerprint(model)}, "backend": "test"}
    artifact = synthesize_rules(model, candidates, rows, causal_artifact=evidence)
    artifact.validate()
    assert artifact.nodes[1].interpretation == "opaque"
    assert artifact.nodes[1].weights
    assert artifact.provenance["causal_evidence"] == evidence


def test_compiler_budgets_and_three_domain_separation():
    from weightscript.synthesis import cegis, compile_rule
    model = tiny_model()
    rule = {"op": "token_at", "index": 0, "default": 0}
    rows = [{"input_ids": [1, 2]}]
    with pytest.raises(ValueError, match="budget"):
        compile_rule(model, rule, rows, max_variables=1)
    with pytest.raises(ValueError, match="overlap"):
        cegis(model, rule, rows, rows)


def test_selection_min_max_and_span_programs():
    ids = [1, 7, 3, 9, 2]
    base = {"op": "select_token", "token": 4, "comparison": "gt", "which": "min", "default": 0}
    assert rule_predict(base, ids) == 7
    assert rule_predict(dict(base, which="max"), ids) == 9
    assert rule_predict(dict(base, comparison="lt", which="max", ignore_tokens=[1]), ids) == 3
    assert rule_predict(dict(base, token=100), ids) == 0
    with pytest.raises(ValueError, match="min or max"):
        validate_rule(dict(base, which="middle"))
    copy_rule = {"op": "span_next", "kind": "copy", "bos": 8, "sep": 9, "eos": 7}
    assert rule_predict(copy_rule, [8, 4, 1, 9]) == 4
    assert rule_predict(copy_rule, [8, 4, 1, 9, 4, 1]) == 7
    assert rule_predict(dict(copy_rule, kind="sorted"), [8, 4, 1, 9]) == 1
    majority = dict(copy_rule, kind="majority")
    assert rule_predict(majority, [8, 0, 0, 1, 9]) == 0
    assert rule_predict(majority, [8, 1, 1, 0, 9]) == 1
    from weightscript.backends.rules import supervised_pairs
    assert supervised_pairs([8, 4, 1, 9, 4], 9) == [([8, 4, 1, 9], 4)]
    for bad in (dict(copy_rule, kind="reverse"), dict(copy_rule, bos=9),
                dict(copy_rule, eos=9)):
        with pytest.raises(ValueError):
            validate_rule(bad)
    with pytest.raises(ValueError, match="outside"):
        rule_predict(copy_rule, [8, 4, 9, 1])
    with pytest.raises(ValueError, match="undefined"):
        rule_predict(majority, [8, 0, 1, 9])


def test_sequence_synthesis_keeps_final_domain_out_of_selection():
    from weightscript.core import fingerprint
    from weightscript.symbolic import synthesize_sequence_rules
    model = tiny_model()
    source = fingerprint(model)
    markers = {"bos": 0, "sep": 4, "eos": 3}
    candidates = [{"op": "span_next", "kind": kind, **markers} for kind in ("copy", "sorted", "majority")]
    calibration = [{"input_ids": [0, 1, 2, 4, 1, 2, 3]}]
    evaluation = [{"input_ids": [0, 2, 1, 4, 2, 1, 3]}]
    final = [{"input_ids": [0, 1, 1, 4, 1]}]
    artifact = synthesize_sequence_rules(model, candidates, calibration, evaluation, final)
    artifact.validate()
    assert fingerprint(model) == source
    assert artifact.nodes[1].interpretation == "opaque"
    assert artifact.fidelity["final"]["positions"] == 1
    assert artifact.fidelity["calibration"]["positions"] == 3
    assert artifact.provenance["causal_evidence"] is None
    # Recompute calibration agreements independently. Final rows must not change the winner.
    import torch
    from weightscript.backends.rules import rule_predict
    logits = model(torch.tensor([calibration[0]["input_ids"]]), use_cache=False).logits[0]
    scores = []
    for rule in candidates:
        correct = 0
        for index in (4, 5, 6):
            try:
                actual = rule_predict(rule, calibration[0]["input_ids"][:index])
            except ValueError:
                actual = None
            correct += actual == int(logits[index - 1].argmax())
        scores.append(correct)
    expected = max(range(3), key=lambda i: (scores[i], -i))
    assert artifact.fidelity["selected_index"] == expected
    with pytest.raises(ValueError, match="overlap"):
        synthesize_sequence_rules(model, candidates, calibration, calibration)
    with pytest.raises(ValueError, match="span_next"):
        synthesize_sequence_rules(model, [{"op": "token_at", "index": 0, "default": 0}], calibration)
    with pytest.raises(ValueError, match="different checkpoint"):
        synthesize_sequence_rules(model, candidates, calibration, causal_artifact={})


def test_min_selection_and_span_copy_compile_without_mutating_source():
    pytest.importorskip("z3")
    import torch
    from weightscript.core import fingerprint
    from weightscript.synthesis import compile_rule
    model = tiny_model()
    source = fingerprint(model)
    edited, report = compile_rule(
        model, {"op": "select_token", "token": 0, "comparison": "gt", "which": "min", "default": 0},
        [{"input_ids": [3, 1, 2]}, {"input_ids": [2, 4, 3]}, {"input_ids": [4, 2, 1]}], timeout=20)
    assert report["passed"]
    assert edited(torch.tensor([[3, 1, 2]])).logits[:, -1].argmax().item() == 1
    span, span_report = compile_rule(
        model, {"op": "span_next", "kind": "copy", "bos": 0, "sep": 1, "eos": 2},
        [{"input_ids": [0, 3, 1]}, {"input_ids": [0, 3, 1, 3]}, {"input_ids": [0, 4, 1]}], timeout=20)
    assert span_report["passed"]
    assert span(torch.tensor([[0, 4, 1]])).logits[:, -1].argmax().item() == 4
    assert fingerprint(model) == source


def test_cegis_rescores_original_oracle_without_mutating_source():
    pytest.importorskip("z3")
    from weightscript.core import fingerprint
    from weightscript.synthesis import cegis
    model = tiny_model()
    source = fingerprint(model)
    rule = {"op": "token_at", "index": 0, "default": 0}
    edited, report = cegis(
        model, rule, [{"input_ids": [1, 4]}, {"input_ids": [2, 4]}],
        [{"input_ids": [3, 4]}], heldout=[{"input_ids": [0, 4]}], max_rounds=2)
    assert fingerprint(model) == source
    assert report["oracle_rescore"]["checked"] == 1
    assert report["heldout_decisions"]["checked"] == 1
    assert edited is not None
    assert "oracle_rows_absorbed" in report


def test_stage_summary_rejects_other_backends():
    from weightscript.backends.pipeline import summarize_stages
    with pytest.raises(ValueError, match="three-stage"):
        summarize_stages({"backend": "other"})


def test_public_domain_validation():
    from weightscript.backends.pipeline import disjoint_domains
    assert len(disjoint_domains(synthesis=[{"input_ids": [1]}], final=[{"input_ids": [2]}])) == 2
    for domains in ({"synthesis": []},
                    {"synthesis": [{"input_ids": [True]}]},
                    {"synthesis": [{"input_ids": [1]}], "final": [{"input_ids": [1]}]}):
        with pytest.raises(ValueError):
            disjoint_domains(**domains)
