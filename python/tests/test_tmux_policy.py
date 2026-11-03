import json
import pytest
from weightscript import tmux_policy


def test_file_bridge_uses_actual_response_and_refuses_stale_files(
    tmp_path, monkeypatch
):
    request = {"step": 0, "context": {"mode": "direct"}, "history": []}

    def responder(session, message):
        assert session == "test-editor"
        assert "request-0000.json" in message
        assert json.loads((tmp_path / "request-0000.json").read_text()) == request
        (tmp_path / "response-0000.json").write_text('{"action":"finish"}')

    monkeypatch.setattr(tmux_policy, "send", responder)
    result = tmux_policy.exchange(tmp_path, "test-editor", request, timeout=1)
    assert result["action"] == "finish"
    assert result["usage"]["policy"] == "codex-tui"
    with pytest.raises(ValueError, match="fresh workspace"):
        tmux_policy.exchange(tmp_path, "test-editor", request, timeout=1)


def test_bridge_rejects_invalid_reply(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tmux_policy,
        "send",
        lambda session, message: (tmp_path / "response-0001.json").write_text("[]"),
    )
    with pytest.raises(ValueError, match="must be an object"):
        tmux_policy.exchange(tmp_path, "test", {"step": 1}, timeout=1)
