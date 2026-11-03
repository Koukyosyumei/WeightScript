import json

import pytest

from weightscript.agent_bridge import command_for, discover
from weightscript import tmux_policy


def test_grok_command_uses_flags_from_the_installed_help():
    spec = discover("grok")
    argv = command_for(
        spec,
        workspace="/tmp/weightscript-agent-workspace",
        session_id="00000000-0000-4000-8000-000000000001",
        model="grok-4",
    )
    assert argv[0] == spec["executable"]
    assert spec["version"]
    flags = [arg for arg in argv if arg.startswith("--")]
    for flag in flags:
        assert flag in spec["help"]
    assert "--leader-socket" in argv
    assert "/tmp/weightscript-agent-workspace/.leader.sock" in argv
    assert "grok-4" in argv


def test_grok_command_fails_closed_without_required_help_flags():
    with pytest.raises(ValueError, match="help"):
        command_for(
            {"executable": "grok", "help": "Usage: grok\n", "version": "test"},
            workspace="/tmp/ws",
            session_id="00000000-0000-4000-8000-000000000001",
        )


def test_grok_bridge_labels_real_file_responses(tmp_path, monkeypatch):
    request = {"step": 2, "request_nonce": "abc", "run_id": "run"}

    def responder(session, message):
        assert "request-0002.json" in message
        (tmp_path / "response-0002.json").write_text(
            json.dumps({"action": "finish", "request_nonce": "abc"})
        )

    monkeypatch.setattr(tmux_policy, "send", responder)
    result = tmux_policy.exchange(
        tmp_path, "editor", request, timeout=1, provider="grok"
    )
    assert result["action"] == "finish"
    assert result["usage"]["policy"] == "grok-tui"
    assert result["usage"]["token_metering"].startswith("not measured")
