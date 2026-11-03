"""Durable session journal, policy-response cache, and thin model snapshots."""

from __future__ import annotations

import functools
import inspect
import json
import os
import uuid
from pathlib import Path

from .storage import atomic_json, controller_lock, digest


def exclusive_output(function):
    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        output = bound.arguments.get("output")
        if output is None:
            return function(*args, **kwargs)
        root = Path(output)
        if (
            not bound.arguments.get("resume", False)
            and root.exists()
            and any(p.name != ".controller.lock" for p in root.iterdir())
        ):
            raise ValueError(
                "session output must be empty; use resume=True to recover this run"
            )
        with controller_lock(root):
            return function(*args, **kwargs)

    return wrapped


class SessionState:
    def __init__(self, output, configuration, resume=False):
        self.root = Path(output) if output is not None else None
        self.path = self.root / "state.json" if self.root else None
        if resume:
            if self.path is None or not self.path.exists():
                raise ValueError("no saved session state to resume")
            self.data = json.loads(self.path.read_text())
            if self.data.get("format") != "weightscript.session.v2" or self.data[
                "configuration_digest"
            ] != digest(configuration):
                raise ValueError(
                    "session source, data, goal, policy or budget differs from saved run"
                )
        else:
            self.data = {
                "format": "weightscript.session.v2",
                "run_id": uuid.uuid4().hex,
                "configuration_digest": digest(configuration),
                "configuration": configuration,
                "spent": 0,
                "history": [],
                "next_step": 0,
                "baseline": None,
                "best_metrics": None,
                "best_proposal": {},
                "candidate_snapshot": None,
                "best_snapshot": None,
                "pending_request": None,
                "pending_response": None,
                "selection_complete": False,
                "final_metrics": {},
                "completed": False,
            }
            self.save()
        self.nonce = uuid.uuid4().hex
        self.data.setdefault("controllers", [])
        self.data["controllers"].append(
            {
                "nonce": self.nonce,
                "pid": os.getpid(),
                "resume": bool(resume),
            }
        )
        self.data["active_controller"] = self.nonce
        self.save()

    def save(self):
        if self.path:
            atomic_json(self.path, self.data)

    def snapshot(self, original, edited):
        if self.root is None:
            return None
        from .patches import save_delta

        relative = Path("snapshots") / uuid.uuid4().hex
        save_delta(original, edited, self.root / relative)
        return str(relative)

    def restore(self, original, reference):
        if not reference:
            return original
        from .patches import apply_delta

        path = (self.root / reference).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError("invalid snapshot reference")
        return apply_delta(original, path)
