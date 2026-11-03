"""CLI wiring for the three-stage driver and explicit semantic rules."""

import json

from weightscript.workbench import main


def test_rule_commands_execute_without_a_checkpoint(tmp_path, capsys):
    rule = {
        "op": "span_next",
        "kind": "majority",
        "bos": 8,
        "sep": 9,
        "eos": 7,
    }
    path = tmp_path / "rule.json"
    path.write_text(json.dumps(rule))
    assert main(["rule-check", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["op"] == "span_next"
    assert main(["rule-predict", str(path), "[8,1,1,0,9]"]) == 0
    assert json.loads(capsys.readouterr().out)["token"] == 1
    path.write_text('{"op": "span_next", "kind": "average"}')
    assert main(["rule-check", str(path)]) == 1


def test_three_stage_cli_uses_the_stage_driver(monkeypatch, tmp_path, capsys):
    class Artifact:
        def to_dict(self):
            return {"backend": "decompiling_transformers.three_stage"}

    seen = {}

    def stages(model, checkout, rows, **kwargs):
        seen["checkout"] = checkout
        seen["rows"] = rows
        seen["kwargs"] = kwargs
        return Artifact()

    monkeypatch.setattr("weightscript.core.load_model", lambda source: object())
    monkeypatch.setattr(
        "weightscript.backends.upstream_stages.decompile_stages", stages
    )
    monkeypatch.setattr(
        "weightscript.backends.pipeline.summarize_stages",
        lambda artifact: {"accepted": False, "stages": []},
    )
    monkeypatch.setattr(
        "weightscript.backends.upstream.decompile",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("component")),
    )
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text('{"input_ids": [2, 0, 3]}\n')
    output = tmp_path / "out"
    assert (
        main(
            [
                "decompile",
                "unused-model",
                str(calibration),
                str(output),
                "--checkout",
                "upstream-checkout",
                "--engine",
                "three-stage",
            ]
        )
        == 0
    )
    assert seen["checkout"] == "upstream-checkout"
    assert seen["rows"] == [{"input_ids": [2, 0, 3]}]
    assert seen["kwargs"]["calibration_steps"] == 32
    assert seen["kwargs"]["max_trials"] == 64
    summary = json.loads((output / "summary.json").read_text())
    assert summary["accepted"] is False
    assert not (output / "program.txt").exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["accepted"] is False


def test_component_engine_rejects_stage_only_flags(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("weightscript.core.load_model", lambda source: object())
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text('{"input_ids": [1]}\n')
    code = main(
        [
            "decompile",
            "unused-model",
            str(calibration),
            str(tmp_path / "out"),
            "--checkout",
            "upstream-checkout",
            "--calibration-steps",
            "4",
        ]
    )
    assert code == 1
    assert "three-stage" in capsys.readouterr().err


def test_qk_replay_uses_the_stage_executor(monkeypatch, tmp_path, capsys):
    import torch

    called = {}

    def execute_stages(model, artifact, checkout, ids, stage):
        called["stage"] = stage
        called["artifact"] = artifact
        return torch.zeros(1, len(ids[0]), 3)

    monkeypatch.setattr("weightscript.core.load_model", lambda source: object())
    monkeypatch.setattr(
        "weightscript.backends.upstream_stages.execute_stages", execute_stages
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"backend": "decompiling_transformers.three_stage"}')
    assert (
        main(
            [
                "run-decompiled",
                "unused-model",
                str(artifact),
                "[4,5]",
                "--checkout",
                "upstream-checkout",
                "--stage",
                "qk",
            ]
        )
        == 0
    )
    assert called["stage"] == "qk"
    assert json.loads(capsys.readouterr().out)["predicted_tokens"] == [[0, 0]]


def test_sequence_synthesis_requires_explicit_candidates(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("weightscript.core.load_model", lambda source: object())
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text('{"input_ids": [1, 2]}\n')
    assert (
        main(
            [
                "synthesize",
                "unused-model",
                str(calibration),
                str(tmp_path / "out"),
                "--sequence",
            ]
        )
        == 1
    )
    assert "candidates" in capsys.readouterr().err
