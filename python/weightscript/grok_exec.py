"""One-turn Grok policy. The model reply and usage come from the Grok CLI.

This does not start Codex, scrape a terminal, or replace a failed reply with an edit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path

from .agent_bridge import discover


def _require(help_text, flags):
    missing = [flag for flag in flags if flag not in help_text]
    if missing:
        raise ValueError("grok --help is missing flags: " + ", ".join(missing))


def _json_object(text):
    """Parse one JSON object. A single fenced block is accepted; prose is not."""
    if not isinstance(text, str):
        raise ValueError("grok text was not a string")
    body = text.strip()
    if body.startswith("```"):
        lines = body.splitlines()
        if len(lines) < 3 or not lines[-1].startswith("```"):
            raise ValueError("grok text was not a JSON action")
        body = "\n".join(lines[1:-1]).strip()
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError("grok action must be an object")
    return body


def ask(request, workspace, timeout=180):
    if not isinstance(request, dict):
        raise ValueError("policy request must be an object")
    spec = discover("grok")
    _require(
        spec["help"],
        (
            "--single",
            "--output-format",
            "--no-subagents",
            "--disable-web-search",
            "--max-turns",
            "--leader-socket",
            "--cwd",
        ),
    )
    root = Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    prompt = (
        "You are a WeightScript editing policy. The next block is the controller "
        "request. Choose exactly one action object as described by context.actions "
        "and context.action_examples. Return only that JSON object. Copy "
        "request_nonce and run_id from the request when they are present. Do not "
        "run model inference, training, or tools.\n"
        + json.dumps(request, allow_nan=False)
    )
    argv = [
        spec["executable"],
        "--single",
        prompt,
        "--output-format",
        "json",
        "--no-subagents",
        "--disable-web-search",
        "--max-turns",
        "1",
        "--leader-socket",
        str(root / f".leader-{uuid.uuid4().hex}.sock"),
        "--cwd",
        str(root),
    ]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError("grok policy timed out") from error
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "grok failed").strip()
        raise ValueError("grok policy failed: " + detail[:2000])
    try:
        wrapper = json.loads(proc.stdout)
    except ValueError as error:
        raise ValueError("grok did not return a JSON wrapper") from error
    try:
        action = json.loads(_json_object(wrapper["text"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("grok text was not a JSON action") from error
    if not isinstance(action, dict):
        raise ValueError("grok action must be an object")
    usage = wrapper.get("usage") if isinstance(wrapper.get("usage"), dict) else {}
    action["usage"] = {
        "provider": "grok-exec",
        "session_id": wrapper.get("sessionId"),
        "model_usage": wrapper.get("modelUsage"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "total_cost_usd": wrapper.get("total_cost_usd"),
        "scope": "verbatim fields from grok --output-format json for this turn",
    }
    return action


def main(argv=None):
    parser = argparse.ArgumentParser(description="Grok one-turn WeightScript policy")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args(argv)
    action = ask(json.load(sys.stdin), args.workspace, args.timeout)
    print(json.dumps(action))


if __name__ == "__main__":
    main()
