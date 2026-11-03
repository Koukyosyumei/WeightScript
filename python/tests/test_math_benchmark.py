"""Edge cases for the math reproduction harness.

These tests do not mirror every helper. They cover extraction limits, the
refusal to replace math-verify, split isolation, resume, and comparison.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from weightscript.math_benchmark import (
    INSTRUCTION,
    Budgets,
    ConfigMismatch,
    IntegrityError,
    ProtocolError,
    RunSpec,
    SplitIsolationError,
    assert_disjoint,
    compare_runs,
    extract_answer,
    grade_response,
    is_equiv,
    load_jsonl,
    loop_signals,
    protocol,
    render_chat,
    run_evaluation,
    user_turn,
)
from weightscript.math_benchmark import extract as extract_module


ROOT = Path(__file__).resolve().parents[2]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _examples(tmp_path: Path, rows=None, *, dataset="fixture", split="test", role="evaluation"):
    rows = rows or [
        {"id": "a", "problem": "What is 20+4?", "answer": "24"},
        {"id": "b", "problem": "What is 3+4?", "answer": "7"},
    ]
    path = tmp_path / "problems.jsonl"
    _write_jsonl(path, rows)
    return load_jsonl(path, dataset=dataset, split=split, role=role)


def _spec(arm="untouched", checkpoint="base", fingerprint=None) -> RunSpec:
    return RunSpec(
        source_model="Qwen/Qwen3.5-0.8B",
        source_revision=None,
        checkpoint_id=checkpoint,
        checkpoint_fingerprint=fingerprint or f"fp-{checkpoint}",
        arm=arm,
        protocol_name="think35",
    )


def _budgets(**kwargs) -> Budgets:
    payload = {"max_samples": 1, "max_new_tokens": 32}
    payload.update(kwargs)
    return Budgets(**payload)


class Clock:
    def __init__(self, values):
        self.values = list(values)
        self.i = 0

    def __call__(self):
        if self.i >= len(self.values):
            return self.values[-1]
        value = self.values[self.i]
        self.i += 1
        return value


@pytest.fixture
def grader_cache():
    saved = dict(extract_module._MV)
    yield extract_module
    extract_module._MV.clear()
    extract_module._MV.update(saved)


def _install_verifier(module, equal):
    class FakeVerifier:
        __version__ = "0.9.0"

        def parse(self, text, parsing_timeout=None):
            return text

        def verify(self, gold, pred, timeout_seconds=None):
            return equal(gold, pred)

    module._MV.update(module=FakeVerifier(), error=None, loaded=True)


def test_import_does_not_load_model_or_grader_packages():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "python")
    env["OMP_NUM_THREADS"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import weightscript.math_benchmark\n"
                "bad = [name for name in ('datasets','math_verify','vllm','torch','transformers') "
                "if name in sys.modules]\n"
                "sys.stderr.write(','.join(bad))\n"
                "sys.exit(1 if bad else 0)\n"
            ),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""


def test_boxed_extraction_follows_the_published_limits():
    assert user_turn("What is 2?") == "What is 2?\n" + INSTRUCTION
    assert extract_answer("<think>\\boxed{1}</think>\nTherefore \\boxed{2}")[0] == "2"
    assert extract_answer("\\boxed{1} \\boxed{2")[0] is None
    assert extract_answer("\\boxed{1} \\fbox{9}")[0] == "1"
    assert extract_answer("The value is \\boxed 5$ more")[0] == "5"
    assert extract_answer("\\boxed{\\frac{1}{2}}")[0] == "\\frac{1}{2}"
    assert extract_answer("no box here")[0] is None
    assert is_equiv("0.5", "\\frac{1}{2}")
    assert is_equiv("1/2", "\\frac{1}{2}")
    assert not is_equiv("2", "3")


def test_missing_math_verify_does_not_fill_correct(grader_cache):
    grader_cache._MV.update(module=None, error=ImportError("math-verify is not installed"), loaded=True)
    graded = grade_response("\\boxed{24}", "24", finish_reason="stop", n_tokens=4)
    assert graded["equiv"] is True
    assert graded["correct"] is None
    assert graded["grader"] == "unavailable"
    unboxed = grade_response("I never boxed it.", "24", finish_reason="stop", n_tokens=4)
    assert unboxed["correct"] is False
    assert unboxed["grader"] == "not_applicable"


def test_verifier_signature_mismatch_stays_ungraded(grader_cache):
    class Broken:
        __version__ = "0.9.0"

        def parse(self, *args, **kwargs):
            raise TypeError("unexpected keyword")

        def verify(self, *args, **kwargs):
            raise AssertionError("verify should not run")

    grader_cache._MV.update(module=Broken(), error=None, loaded=True)
    graded = grade_response("\\boxed{24}", "24", finish_reason="length", n_tokens=8)
    assert graded["correct"] is None
    assert graded["equiv"] is True
    assert "signature" in graded["grader_error"]
    assert graded["cut"] is True


def test_loop_signals_and_token_cap_are_distinct():
    assert loop_signals("Wait " * 39)["verbatim_loop_end"] is False
    assert loop_signals("one two three")["repetitive_end"] is False
    forty = loop_signals("Wait " * 40)
    assert forty["verbatim_loop_end"] is False
    assert forty["repetitive_end"] is True
    forty_one = loop_signals("Wait " * 41)
    assert forty_one["verbatim_loop_end"] is True
    assert forty_one["repetitive_end"] is True
    assert loop_signals("The car stays on the ramp.")["verbatim_loop_end"] is False
    assert grade_response("done", "1", finish_reason="length")["cut"] is True
    assert grade_response("done", "1", finish_reason="stop")["cut"] is False
    assert grade_response("done", "1", finish_reason=None)["cut"] is None


def test_split_isolation_and_gsm8k_gold(tmp_path):
    raw = tmp_path / "gsm.jsonl"
    _write_jsonl(
        raw,
        [{"id": "g0", "problem": "How many?", "answer": "Natalia sold clips.\n#### 1,024"}],
    )
    evaluation = load_jsonl(raw, dataset="gsm8k", split="test", role="evaluation")
    assert evaluation.examples[0].answer == "1024"
    assert evaluation.provenance["file_sha256"]
    assert evaluation.provenance["article_13gram_decontamination"] == "not_applied_filter_unpublished"
    math_rows = tmp_path / "math.jsonl"
    _write_jsonl(math_rows, [{"id": "m0", "problem": "Tuple?", "answer": "1,000"}])
    math500 = load_jsonl(math_rows, dataset="math500", split="test", role="evaluation")
    assert math500.examples[0].answer == "1,000"
    with pytest.raises(SplitIsolationError):
        load_jsonl(raw, dataset="gsm8k", split="train", role="evaluation")
    with pytest.raises(SplitIsolationError):
        load_jsonl(math_rows, dataset="math500", split="test", role="practice")
    practice_path = tmp_path / "practice.jsonl"
    _write_jsonl(practice_path, [{"id": "t0", "problem": "How many?", "answer": "#### 5"}])
    practice = load_jsonl(practice_path, dataset="gsm8k", split="train", role="practice")
    with pytest.raises(SplitIsolationError, match="13-word"):
        assert_disjoint(evaluation, practice)
    other = load_jsonl(
        _touch(tmp_path / "other.jsonl", [{"id": "t1", "problem": "Unrelated?", "answer": "#### 5"}]),
        dataset="gsm8k",
        split="train",
        role="practice",
    )
    assert_disjoint(evaluation, other)
    with pytest.raises(SplitIsolationError):
        run_evaluation(practice, _spec(), lambda requests: [], tmp_path / "no.jsonl", budgets=_budgets())


def _touch(path: Path, rows: list[dict]) -> Path:
    _write_jsonl(path, rows)
    return path


def test_chat_template_requires_enable_thinking():
    class Good:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=None):
            assert enable_thinking is True
            assert tokenize is False
            return messages[0]["content"] + "<gen>"

    rendered = render_chat(Good(), "What is 2?", thinking=True)
    assert rendered.endswith(INSTRUCTION + "<gen>")

    class Bad:
        def apply_chat_template(self, *args, **kwargs):
            raise TypeError("enable_thinking")

    with pytest.raises(ProtocolError, match="enable_thinking"):
        render_chat(Bad(), "What is 2?", thinking=True)


def test_requests_hide_gold_and_accuracy_is_a_mean(tmp_path, grader_cache):
    _install_verifier(grader_cache, lambda gold, pred: gold == pred)
    examples = _examples(tmp_path, dataset="math500")
    seen = []

    def generate(requests):
        seen.extend(requests)
        assert requests[0]["sampling"]["temperature"] == 0.6
        assert requests[0]["sampling"]["presence_penalty"] == 1.5
        assert requests[0]["sampling"]["top_p"] == 0.95
        assert requests[0]["sampling"]["top_k"] == 20
        assert requests[0]["sampling"]["max_tokens"] == 32
        assert requests[0]["seed"] == 0
        assert requests[0]["user_content"].endswith(INSTRUCTION)
        return [
            {"text": "\\boxed{24}", "finish_reason": "stop", "n_tokens": 4},
            {"text": "\\boxed{0}", "finish_reason": "stop", "n_tokens": 6},
        ]

    summary = run_evaluation(
        examples,
        _spec(),
        generate,
        tmp_path / "run.jsonl",
        budgets=_budgets(max_examples=1, max_samples=2),
    )
    assert {key for request in seen for key in ("gold", "answer", "solution") if key in request} == set()
    assert summary["n_problems"] == 1
    assert summary["acc"] == 0.5
    assert summary["headline"] == "acc"
    assert summary["mean_tokens"] == 5
    assert summary["cut_share"] == 0
    assert "below the published 4" in " ".join(summary["deviations"])
    assert summary["metrics_are_article_grader"] is True
    report = compare_runs(tmp_path / "run.jsonl", tmp_path / "run.jsonl")
    assert report["blog_table_settings"] is False


def test_resume_skips_completed_examples_and_rejects_a_different_config(tmp_path, grader_cache):
    _install_verifier(grader_cache, lambda gold, pred: gold == pred)
    examples = _examples(tmp_path)
    calls = []

    def generate(requests):
        example_id = requests[0]["example_id"]
        calls.append(example_id)
        if example_id == "b" and calls.count("b") == 1:
            raise RuntimeError("stop after the first example")
        return [{"text": "\\boxed{24}", "finish_reason": "stop", "n_tokens": 2}]

    out = tmp_path / "resume.jsonl"
    with pytest.raises(RuntimeError, match="stop after"):
        run_evaluation(examples, _spec(), generate, out, budgets=_budgets())
    assert calls == ["a", "b"]
    run_evaluation(examples, _spec(), generate, out, budgets=_budgets())
    assert calls == ["a", "b", "b"]
    stored = out.read_bytes()
    with pytest.raises(ConfigMismatch, match="checkpoint_id"):
        run_evaluation(examples, _spec(checkpoint="other"), generate, out, budgets=_budgets())
    assert out.read_bytes() == stored
    out.write_bytes(stored + b'{"record":"sample","cut":')
    calls.clear()
    run_evaluation(examples, _spec(), generate, out, budgets=_budgets())
    assert calls == []
    assert out.read_bytes().endswith(b"\n")
    assert b'"cut":' not in out.read_bytes().splitlines()[-1]


def test_non_trailing_hole_is_rejected_without_rewriting(tmp_path, grader_cache):
    _install_verifier(grader_cache, lambda gold, pred: True)
    examples = _examples(tmp_path)
    out = tmp_path / "hole.jsonl"

    def generate(requests):
        return [
            {"text": "\\boxed{1}", "finish_reason": "stop", "n_tokens": 1}
            for _request in requests
        ]

    run_evaluation(examples, _spec(), generate, out, budgets=_budgets(max_samples=2))
    lines = out.read_text().splitlines()
    sample_lines = [line for line in lines if '"sample_index":1' in line and '"example_id":"a"' in line]
    assert sample_lines
    damaged = "\n".join(line for line in lines if line not in sample_lines) + "\n"
    before = damaged.encode()
    out.write_bytes(before)
    with pytest.raises(IntegrityError, match="unfinished"):
        run_evaluation(examples, _spec(), generate, out, budgets=_budgets(max_samples=2))
    assert out.read_bytes() == before


def test_walltime_stops_between_examples_and_accumulates(tmp_path):
    examples = _examples(tmp_path)
    calls = []

    def generate(requests):
        calls.append(requests[0]["example_id"])
        return [{"text": "no box", "finish_reason": "stop", "n_tokens": 2}]

    out = tmp_path / "wall.jsonl"
    first = run_evaluation(
        examples,
        _spec(),
        generate,
        out,
        budgets=_budgets(walltime_s=15),
        clock=Clock([0, 0, 100, 100]),
    )
    assert calls == ["a"]
    assert first["complete"] is False
    assert first["stopped_reason"] == "walltime"
    run_evaluation(
        examples,
        _spec(),
        generate,
        out,
        budgets=_budgets(walltime_s=15),
        clock=Clock([0, 0, 0]),
    )
    assert calls == ["a"]


def test_bad_finish_is_not_stored(tmp_path):
    examples = _examples(tmp_path)
    out = tmp_path / "bad.jsonl"

    def generate(requests):
        return [{"text": "\\boxed{1}", "finish_reason": "abort", "n_tokens": 1}]

    with pytest.raises(Exception, match="finish_reason"):
        run_evaluation(examples, _spec(), generate, out, budgets=_budgets(max_examples=1))
    text = out.read_text()
    assert "abort" not in text
    assert '"record":"sample"' not in text


def test_comparison_requires_equal_full_sets(tmp_path, grader_cache):
    _install_verifier(grader_cache, lambda gold, pred: gold == pred)
    examples = _examples(tmp_path)
    answers = {"a": "24", "b": "7"}

    def make(arm, checkpoint, box):
        def generate(requests):
            example_id = requests[0]["example_id"]
            for request in requests:
                assert "gold" not in request
            text = "\\boxed{" + (answers[example_id] if box == "right" else "0") + "}"
            finish = "length" if example_id == "b" and box == "right" else "stop"
            return [{"text": text, "finish_reason": finish, "n_tokens": 4}]

        path = tmp_path / f"{arm}.jsonl"
        run_evaluation(examples, _spec(arm=arm, checkpoint=checkpoint), generate, path, budgets=_budgets())
        return path

    untouched = make("untouched", "base", "wrong")
    edited = make("edited", "edit", "right")
    report = compare_runs(untouched, edited)
    assert report["comparable"] is True
    assert report["complete"] is True
    assert report["delta_acc"] == 1
    assert report["blog_table_settings"] is False
    assert any("local JSONL" in item for item in report["blog_table_blockers"])
    assert report["untouched"]["cut_share"] == 0
    assert report["edited"]["cut_share"] == 0.5
    assert [row["edited_correct"] for row in report["per_example"]] == [[True], [True]]

    shorter = _budgets(max_new_tokens=16)
    other = tmp_path / "shorter.jsonl"
    run_evaluation(
        examples,
        _spec(arm="edited", checkpoint="edit-short"),
        lambda requests: [{"text": "\\boxed{0}", "finish_reason": "stop", "n_tokens": 1}],
        other,
        budgets=shorter,
    )
    mismatched = compare_runs(untouched, other)
    assert mismatched["comparable"] is False
    assert mismatched["delta_acc"] is None
    assert any("max_new_tokens" in reason for reason in mismatched["incomparable_because"])

    lines = edited.read_text().splitlines()
    kept = [line for line in lines if '"example_id":"a"' not in line]
    selected = tmp_path / "selected.jsonl"
    selected.write_text("\n".join(kept) + "\n")
    picked = compare_runs(untouched, selected)
    assert picked["comparable"] is False
    assert picked["delta_acc"] is None
    assert any("canonical prefix" in reason for reason in picked["incomparable_because"])


def test_published_protocol_objects_match_the_evaluator_constants():
    think = protocol("think35")
    assert think.sampling == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
        "max_tokens": 32768,
        "seed": 0,
    }
    assert think.thinking is True
    assert think.samples == {"math500": 4, "gsm8k": 2}
    nothink = protocol("nothink35")
    assert nothink.thinking is False
    assert nothink.sampling["temperature"] == 0.7
    assert nothink.sampling["top_p"] == 0.8
    assert nothink.sampling["max_tokens"] == 8192
    gemma = protocol("gemma3")
    assert "presence_penalty" not in gemma.sampling
    assert gemma.sampling["temperature"] == 1.0
    assert gemma.sampling["top_k"] == 64
    assert gemma.lenient_headline is True
    assert gemma.engine is None


def test_sample_budget_cannot_exceed_the_published_count(tmp_path):
    examples = _examples(
        tmp_path,
        [{"id": "m0", "problem": "What is 1+1?", "answer": "2"}],
        dataset="math500",
    )
    with pytest.raises(ProtocolError, match="exceeds the published"):
        run_evaluation(
            examples,
            _spec(),
            lambda requests: [],
            tmp_path / "out.jsonl",
            budgets=Budgets(max_samples=8),
        )


def test_smoke_command_resumes_and_does_not_claim_the_blog_table(tmp_path):
    script = ROOT / "examples" / "math_benchmark.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "python")
    env["OMP_NUM_THREADS"] = "1"
    completed = subprocess.run(
        [sys.executable, str(script), "smoke", "--out", str(tmp_path / "smoke")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["generator_calls_for_untouched"] == 2
    assert payload["blog_table_settings"] is False
    assert payload["comparable"] is True
    assert payload["complete"] is True
    assert "local JSONL" in " ".join(payload["blog_table_blockers"])
