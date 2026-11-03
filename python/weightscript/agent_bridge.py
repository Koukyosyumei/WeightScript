"""Provider-neutral start and usage helpers for external editing agents.

The file bridge lives in tmux_policy.exchange. This module discovers the Grok CLI
from its real help text and can start that CLI in tmux. It does not log in, invent
a model reply, or scrape a READY message.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path

REQUIRED_GROK_FLAGS = (
    "--no-alt-screen",
    "--cwd",
    "--leader-socket",
    "--session-id",
)

GROK_INSTRUCTIONS = """You are an independent WeightScript editing agent in a Grok session.
Your target is a pretrained local model, not the WeightScript framework source.
Work only in this directory. The host sends request-NNNN.json files with goals,
development observations, mode-specific tool examples and budgets. Do not access
other sessions, checkpoints, benchmark files, credentials, or the parent repo.
Do not run target-model inference yourself, train, use gradients, or spawn agents.
Read each request and write exactly one action object to response-NNNN.json
atomically (temporary file, then rename). Copy request_nonce and run_id when the
request includes them. The host executes inspection, traces, generation and
evaluations. Proposals are relative to original weights. After writing a response,
end the turn and wait. Finish with {"action":"finish"}.
"""

START_PROMPT = (
    "Read AGENTS.md in this workspace. You are a new independent editing agent. "
    "Do not reuse any prior experiment. Reply READY and wait for a request file; "
    "do not inspect anything outside this workspace."
)


def discover(provider: str = "grok") -> dict:
    """Run the installed provider CLI help. Fail closed if it is missing."""
    if provider != "grok":
        raise ValueError("automatic CLI discovery is implemented for grok only")
    executable = shutil.which("grok")
    if not executable:
        raise ValueError("grok executable not found on PATH")
    help_run = subprocess.run(
        [executable, "--help"], capture_output=True, text=True, check=False
    )
    if help_run.returncode != 0 or not help_run.stdout:
        raise ValueError(
            "grok --help failed: " + (help_run.stderr or help_run.stdout).strip()
        )
    version = subprocess.run(
        [executable, "--version"], capture_output=True, text=True, check=False
    )
    if version.returncode != 0 or not version.stdout.strip():
        raise ValueError("grok --version failed: " + version.stderr.strip())
    return {
        "provider": "grok",
        "executable": executable,
        "help": help_run.stdout,
        "version": version.stdout.strip(),
    }


def command_for(spec: dict, workspace: str | Path, session_id: str, model=None) -> list:
    """Build a Grok argv using only flags present in that executable's help."""
    help_text = spec.get("help") or ""
    missing = [flag for flag in REQUIRED_GROK_FLAGS if flag not in help_text]
    if missing:
        raise ValueError("grok --help is missing required flags: " + ", ".join(missing))
    if not spec.get("executable"):
        raise ValueError("grok discovery did not return an executable")
    root = Path(workspace)
    argv = [
        spec["executable"],
        "--no-alt-screen",
        "--cwd",
        str(root),
        "--leader-socket",
        str(root / ".leader.sock"),
        "--session-id",
        session_id,
    ]
    if "--no-subagents" in help_text:
        argv.append("--no-subagents")
    if "--disable-web-search" in help_text:
        argv.append("--disable-web-search")
    if model:
        if "--model" not in help_text:
            raise ValueError("grok --help has no --model flag")
        argv.extend(["--model", model])
    if "--permission-mode" in help_text:
        argv.extend(["--permission-mode", "acceptEdits"])
    argv.append(START_PROMPT)
    return argv


def start_grok(workspace, session, model=None) -> dict:
    """Start a real Grok TUI in a detached tmux session. Do not invent its reply."""
    import shlex

    if not shutil.which("tmux"):
        raise ValueError("tmux is required to start a detached agent session")
    spec = discover("grok")
    root = Path(workspace).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("agent workspace must be empty")
    root.mkdir(parents=True, exist_ok=True)
    session_id = str(uuid.uuid4())
    argv = command_for(spec, root, session_id, model)
    (root / "AGENTS.md").write_text(GROK_INSTRUCTIONS)
    record = {
        "provider": "grok",
        "tmux_session": session,
        "session_id": session_id,
        "executable": spec["executable"],
        "version": spec["version"],
        "leader_socket": str(root / ".leader.sock"),
        "authentication": "not checked; the Grok CLI uses its own login state",
        "response_fabricated": False,
    }
    (root / "provider.json").write_text(json.dumps(record, indent=2) + "\n")
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
            shlex.join(argv),
        ],
        check=True,
    )
    subprocess.run(
        ["tmux", "set-option", "-t", session, "destroy-unattached", "off"],
        check=False,
        capture_output=True,
    )
    alive = subprocess.run(["tmux", "has-session", "-t", session], capture_output=True)
    if alive.returncode != 0:
        capture = subprocess.run(
            ["tmux", "capture-pane", "-pt", session, "-S", "-40"],
            capture_output=True,
            text=True,
        )
        detail = (capture.stdout or capture.stderr or "session exited").strip()
        raise ValueError("grok session exited during startup: " + detail[:2000])
    record["status"] = "started"
    record["workspace"] = str(root)
    return record


def read_grok_usage(session_id: str) -> dict:
    """Return the verbatim `grok usage` result. Do not fill in missing totals."""
    spec = discover("grok")
    if "usage" not in spec["help"]:
        raise ValueError("grok --help does not list a usage command")
    proc = subprocess.run(
        [spec["executable"], "usage", session_id],
        capture_output=True,
        text=True,
        check=False,
    )
    parsed = None
    try:
        parsed = json.loads(proc.stdout) if proc.stdout.strip() else None
    except ValueError:
        parsed = None
    return {
        "provider": "grok",
        "session_id": session_id,
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "parsed": parsed,
        "scope": "verbatim output of the installed grok usage command; not an estimated invoice",
    }
