"""Grok policy parsing. The live CLI is not invoked."""

import json

import pytest

from weightscript.grok_exec import ask


class Completed:
    def __init__(self, stdout, returncode=0, stderr=""):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_ask_keeps_verbatim_usage_and_fenced_json(monkeypatch):
    action = {"action": "finish", "request_nonce": "abc", "run_id": "run"}
    wrapper = {
        "text": "```json\n" + json.dumps(action) + "\n```",
        "sessionId": "session-1",
        "modelUsage": {"grok-4.7-build": {}},
        "total_cost_usd": 0.00595816,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 4,
            "total_tokens": 14,
        },
    }
    monkeypatch.setattr(
        "weightscript.grok_exec.discover",
        lambda provider="grok": {
            "executable": "grok",
            "help": "--single --output-format --no-subagents --disable-web-search --max-turns --leader-socket --cwd",
            "version": "grok 1.0.70",
        },
    )
    seen = {}

    def run(argv, **kwargs):
        seen["argv"] = argv
        return Completed(json.dumps(wrapper))

    monkeypatch.setattr("weightscript.grok_exec.subprocess.run", run)
    result = ask({"request_nonce": "abc"}, "workspace", timeout=5)
    assert result["action"] == "finish"
    assert result["usage"]["total_cost_usd"] == 0.00595816
    assert result["usage"]["input_tokens"] == 10
    assert result["usage"]["session_id"] == "session-1"
    assert "--no-subagents" in seen["argv"]
    assert seen["argv"][seen["argv"].index("--max-turns") + 1] == "1"


def test_ask_rejects_non_object_and_failed_process(monkeypatch):
    monkeypatch.setattr(
        "weightscript.grok_exec.discover",
        lambda provider="grok": {
            "executable": "grok",
            "help": "--single --output-format --no-subagents --disable-web-search --max-turns --leader-socket --cwd",
            "version": "grok 1.0.70",
        },
    )

    def run(argv, **kwargs):
        return Completed(json.dumps({"text": "[]"}))

    monkeypatch.setattr("weightscript.grok_exec.subprocess.run", run)
    with pytest.raises(ValueError, match="JSON action"):
        ask({"step": 0}, "workspace")

    def fail(argv, **kwargs):
        return Completed("", returncode=2, stderr="auth missing")

    monkeypatch.setattr("weightscript.grok_exec.subprocess.run", fail)
    with pytest.raises(ValueError, match="auth missing"):
        ask({"step": 0}, "workspace")
