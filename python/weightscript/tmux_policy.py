"""File-based bridge from the policy protocol to an existing interactive Codex TUI.

No terminal scraping or fabricated assistant responses. Codex writes the response
file; the controller validates and executes it through ordinary WeightScript tools.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def send(session, message):
    buffer = f"weightscript-policy-{os.getpid()}"
    # Load a buffer then paste: prompt content cannot become tmux key commands.
    subprocess.run(
        ["tmux", "load-buffer", "-b", buffer, "-"],
        input=message.encode(),
        check=True,
    )
    subprocess.run(
        ["tmux", "paste-buffer", "-b", buffer, "-t", session, "-d"], check=True
    )
    # Let the TUI consume the paste event before submitting it.
    time.sleep(0.25)
    subprocess.run(["tmux", "send-keys", "-t", session, "Enter"], check=True)


def exchange(workspace, session, request, timeout=240):
    root = Path(workspace).resolve()
    if not root.is_dir():
        raise ValueError("Codex workspace does not exist")
    step = request["step"]
    source = root / f"request-{step:04d}.json"
    response = root / f"response-{step:04d}.json"
    if source.exists() or response.exists():
        raise ValueError("use a fresh workspace per agent arm")
    temp = source.with_suffix(".tmp")
    temp.write_text(json.dumps(request, indent=2))
    temp.replace(source)
    message = f"Read AGENTS.md and {source.name}. Choose one tool action for the editing goal and atomically write its JSON to {response.name}. Do not run target-model inference yourself. After writing the file, end this turn and wait for the next request."
    send(session, message)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if response.exists():
            if response.stat().st_size > 1024 * 1024:
                raise ValueError("Codex response exceeds 1 MiB")
            result = json.loads(response.read_text())
            if not isinstance(result, dict):
                raise ValueError("Codex response must be an object")
            result["usage"] = dict(
                result.get("usage", {}),
                policy="codex-tui",
                tmux_session=session,
                token_metering="not available through this file bridge; retain Codex session logs",
            )
            return result
        status = subprocess.run(
            ["tmux", "has-session", "-t", session], capture_output=True
        )
        if status.returncode:
            raise ValueError("Codex tmux session exited before producing a response")
        time.sleep(0.25)
    raise ValueError("Codex did not produce a response before the policy deadline")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--timeout", type=int, default=240)
    args = p.parse_args()
    result = exchange(args.workspace, args.session, json.load(sys.stdin), args.timeout)
    print(json.dumps(result))


INSTRUCTIONS = """You are an independent WeightScript editing agent in a Codex TUI.
Your target is a pretrained local model, not the WeightScript framework source.
Work only in this directory. The host sends request-NNNN.json files with goals,
development observations, mode-specific tool examples and budgets. Do not access
other sessions, checkpoints, benchmark files, credentials, or the parent repo.
Do not run target-model inference yourself, train, use gradients, or spawn agents.
Read each request with python3, inspecting relevant JSON fields rather than
printing enormous graph/weight inventories. Write exactly one action object to
response-NNNN.json atomically (temporary file then rename). The host executes
inspection, traces, generation and evaluations and returns results in the next
request. Proposals are relative to original weights. After writing a response,
end the turn immediately; wait for the host's next prompt. Finish with
{"action":"finish"}. Do not modify framework or experiment code.
"""


def start(workspace, session, model=None):
    import shlex

    if not shutil.which("codex") or not shutil.which("tmux"):
        raise ValueError("codex and tmux must be installed")
    root = Path(workspace).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Codex agent workspace must be empty")
    root.mkdir(parents=True, exist_ok=True)
    (root / "AGENTS.md").write_text(INSTRUCTIONS)
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
    command = [
        "codex",
        "--no-alt-screen",
        "--no-daemon",
        "-a",
        "never",
        "-s",
        "workspace-write",
        "-C",
        str(root),
    ]
    if model:
        command += ["--model", model]
    command.append(
        "Read AGENTS.md in this workspace. You are a new independent editing agent. Do not reuse any prior experiment or thread. Reply READY and wait for a request file; do not inspect anything outside this workspace."
    )
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-x",
            "160",
            "-y",
            "50",
            "-s",
            session,
            "-c",
            str(root),
            shlex.join(command),
        ],
        check=True,
    )
    # Trust only the fresh directory and instructions created by this function.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        capture = subprocess.run(
            ["tmux", "capture-pane", "-pt", session, "-S", "-30"],
            capture_output=True,
            text=True,
        )
        if capture.returncode:
            raise ValueError("Codex TUI exited during startup")
        if "Trust this folder?" in capture.stdout:
            subprocess.run(["tmux", "send-keys", "-t", session, "Enter"], check=True)
            return {
                "session": session,
                "workspace": str(root),
                "trust": "fresh generated workspace",
            }
        if "default ·" in capture.stdout:
            return {
                "session": session,
                "workspace": str(root),
                "trust": "already trusted",
            }
        time.sleep(0.25)
    return {
        "session": session,
        "workspace": str(root),
        "status": "started; inspect tmux pane for readiness",
    }


def read_usage(workspace):
    """Best-effort usage from this workspace's own local Codex rollout only."""
    from datetime import datetime, timezone

    root = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "sessions"
    today = datetime.now(timezone.utc)
    folder = root / today.strftime("%Y/%m/%d")
    results = []
    for path in folder.glob("*.jsonl"):
        try:
            with path.open() as source:
                first = json.loads(next(source))
                meta = first.get("payload", {})
                if meta.get("cwd") != str(Path(workspace).resolve()):
                    continue
                info = None
                model = None
                for line in source:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    payload = event.get("payload", {})
                    if event.get("type") == "turn_context":
                        model = payload.get("model", model)
                    if payload.get("type") == "token_count":
                        info = payload.get("info")
            results.append(
                {
                    "session_id": meta.get("id"),
                    "workspace": meta.get("cwd"),
                    "model": model,
                    "token_usage": info,
                    "scope": "cumulative rollout usage at collection time; includes any debugging or later turns in the same thread",
                }
            )
        except (OSError, ValueError, StopIteration):
            continue
    return results


if __name__ == "__main__":
    main()
