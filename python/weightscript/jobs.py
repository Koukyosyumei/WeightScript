"""Durable, detached command jobs with SQLite state and tmux supervision.

A command retry is explicitly a restart. Stateful WeightScript sessions use their
own recovery journal to resume model/policy work rather than replaying blindly.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .paths import state_directory
from .storage import atomic_json, controller_lock


def process_alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class JobStore:
    def __init__(self, directory: str | Path | None = None):
        self.directory = Path(directory) if directory else state_directory() / "jobs"
        self.directory = self.directory.expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.directory / "jobs.sqlite3", timeout=10)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self._migrate()

    def _migrate(self):
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1, 2):
            raise ValueError(f"unsupported job database version {version}")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS jobs ("
            "id TEXT PRIMARY KEY, status TEXT NOT NULL, argv TEXT NOT NULL, "
            "cwd TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, "
            "pid INTEGER, session TEXT, exit_code INTEGER, "
            "attempts INTEGER NOT NULL DEFAULT 0, child_pid INTEGER, note TEXT)"
        )
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(jobs)")}
        if "child_pid" not in columns:
            self.connection.execute("ALTER TABLE jobs ADD COLUMN child_pid INTEGER")
        if "note" not in columns:
            self.connection.execute("ALTER TABLE jobs ADD COLUMN note TEXT")
        self.connection.execute("PRAGMA user_version=2")
        self.connection.commit()

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def create(self, argv: list[str], cwd: str | Path | None = None) -> str:
        if not argv or any(not isinstance(arg, str) or "\0" in arg for arg in argv):
            raise ValueError("command must be a nonempty argument list")
        path = Path(cwd or Path.cwd()).resolve()
        if not path.is_dir():
            raise ValueError("job working directory does not exist")
        key = uuid.uuid4().hex
        now = time.time()
        self.connection.execute(
            "INSERT INTO jobs(id,status,argv,cwd,created,updated) VALUES(?,?,?,?,?,?)",
            (key, "pending", json.dumps(argv), str(path), now, now),
        )
        self.connection.commit()
        root = self.directory / key
        root.mkdir()
        atomic_json(
            root / "command.json",
            {"id": key, "argv": argv, "cwd": str(path), "created": now},
        )
        return key

    def _row(self, key: str) -> dict:
        row = self.connection.execute(
            "SELECT * FROM jobs WHERE id=?", (key,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown job {key}")
        result = dict(row)
        result["argv"] = json.loads(result["argv"])
        return result

    def _diagnostics(self, job: dict) -> dict:
        session = job.get("session")
        session_alive = None
        if session:
            if shutil.which("tmux"):
                session_alive = (
                    subprocess.run(
                        ["tmux", "has-session", "-t", session], capture_output=True
                    ).returncode
                    == 0
                )
            else:
                session_alive = False
        pid_alive = process_alive(job.get("pid"))
        child_alive = process_alive(job.get("child_pid"))
        notes = []
        if job.get("note"):
            notes.append(job["note"])
        if (
            job["status"] == "running"
            and session
            and session_alive is False
            and pid_alive
        ):
            notes.append(
                "runner is alive but its tmux session is gone; possible orphan after session loss"
            )
        if child_alive and not pid_alive:
            notes.append(
                f"child {job['child_pid']} survived the runner; inspect before retrying"
            )
        return {
            "pid_alive": pid_alive,
            "child_alive": child_alive,
            "tmux_session_alive": session_alive,
            "notes": notes,
        }

    def _mark(self, key: str, status: str, note: str, exit_code=None):
        self.connection.execute(
            "UPDATE jobs SET status=?, note=?, exit_code=?, updated=? WHERE id=?",
            (status, note, exit_code, time.time(), key),
        )
        self.connection.commit()

    def get(self, key: str) -> dict:
        job = self._row(key)
        if job["status"] == "running" and not process_alive(job.get("pid")):
            child = job.get("child_pid")
            if process_alive(child):
                note = (
                    f"runner pid {job.get('pid')} is dead; child {child} is still alive"
                )
            else:
                note = f"runner pid {job.get('pid')} exited without a terminal status"
            self._mark(job["id"], "interrupted", note)
            job = self._row(key)
        job["diagnostics"] = self._diagnostics(job)
        return job

    def list(self) -> list[dict]:
        return [
            self.get(row[0])
            for row in self.connection.execute(
                "SELECT id FROM jobs ORDER BY created DESC"
            ).fetchall()
        ]

    def launch(self, key: str) -> dict:
        if not shutil.which("tmux"):
            raise ValueError("detached jobs require tmux")
        job = self.get(key)
        if job["diagnostics"]["child_alive"]:
            raise ValueError(
                "previous child process is still running; not retrying over an orphan"
            )
        if job["status"] not in {"pending", "failed", "interrupted"}:
            raise ValueError(
                "job is already running or complete; use an explicit new job to rerun"
            )
        session = f"weightscript-job-{key[:12]}"
        if (
            subprocess.run(
                ["tmux", "has-session", "-t", session], capture_output=True
            ).returncode
            == 0
        ):
            raise ValueError(
                "job still has a tmux session; inspect its log before retrying"
            )
        command = [
            sys.executable,
            "-m",
            "weightscript.jobs",
            "--store",
            str(self.directory),
            "_execute",
            key,
        ]
        subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                session,
                "-c",
                job["cwd"],
                shlex.join(command),
            ],
            check=True,
        )
        # A detached session must outlive the client that created it, including
        # an SSH hangup of that client. This is not a request to ignore signals
        # sent to the job itself.
        subprocess.run(
            ["tmux", "set-option", "-t", session, "destroy-unattached", "off"],
            check=False,
            capture_output=True,
        )
        self.connection.execute(
            "UPDATE jobs SET session=?, note=NULL, updated=? WHERE id=?",
            (session, time.time(), key),
        )
        self.connection.commit()
        return self.get(key)

    def execute(self, key: str) -> int:
        root = self.directory / key
        with controller_lock(root):
            job = self._row(key)
            if job["status"] == "succeeded":
                return 0
            if process_alive(job.get("child_pid")):
                raise ValueError(
                    "previous child process is still running; not starting another copy"
                )
            self.connection.execute(
                "UPDATE jobs SET status='running', pid=?, child_pid=NULL, note=NULL, "
                "updated=?, attempts=attempts+1 WHERE id=?",
                (os.getpid(), time.time(), key),
            )
            self.connection.commit()
            process = None
            signaled = {"number": None}

            def forward(signum, _frame):
                signaled["number"] = signum
                current = process
                if current is not None and current.poll() is None:
                    try:
                        current.send_signal(signum)
                    except ProcessLookupError:
                        pass

            previous = {}
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                previous[signum] = signal.signal(signum, forward)
            code = None
            try:
                if signaled["number"] is not None:
                    self._mark(
                        key,
                        "interrupted",
                        f"runner received signal {signaled['number']} before the child started",
                        128 + signaled["number"],
                    )
                    return 128 + signaled["number"]
                with (root / "output.log").open("ab", buffering=0) as log:
                    process = subprocess.Popen(
                        job["argv"],
                        cwd=job["cwd"],
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    self.connection.execute(
                        "UPDATE jobs SET child_pid=?, updated=? WHERE id=?",
                        (process.pid, time.time(), key),
                    )
                    self.connection.commit()
                    while process.poll() is None:
                        if signaled["number"] is not None:
                            try:
                                process.wait(timeout=2)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait(timeout=2)
                            number = signaled["number"]
                            self._mark(
                                key,
                                "interrupted",
                                f"forwarded signal {number} to child {process.pid}",
                                128 + number,
                            )
                            return 128 + number
                        self.connection.execute(
                            "UPDATE jobs SET updated=? WHERE id=?", (time.time(), key)
                        )
                        self.connection.commit()
                        time.sleep(0.2)
                    if signaled["number"] is not None:
                        number = signaled["number"]
                        self._mark(
                            key,
                            "interrupted",
                            f"forwarded signal {number} to child {process.pid}",
                            128 + number,
                        )
                        return 128 + number
                    code = process.returncode
                status = "succeeded" if code == 0 else "failed"
            except BaseException:
                self._mark(key, "interrupted", "runner stopped by an unhandled error")
                raise
            finally:
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
            self.connection.execute(
                "UPDATE jobs SET status=?, exit_code=?, updated=? WHERE id=?",
                (status, code, time.time(), key),
            )
            self.connection.commit()
            atomic_json(root / "result.json", self.get(key))
            return code

    def logs(self, key: str, lines: int = 100) -> str:
        from collections import deque

        self.get(key)
        path = self.directory / key / "output.log"
        if lines < 1:
            raise ValueError("line count must be positive")
        if not path.exists():
            return ""
        with path.open(errors="replace") as stream:
            return "".join(deque(stream, maxlen=lines))


def run_action(args) -> int:
    action = getattr(args, "job_action", None) or getattr(args, "action", None)
    with JobStore(args.store) as store:
        if action == "run":
            command = getattr(args, "job_argv", None)
            if command is None:
                command = args.command
            if command and command[0] == "--":
                command = command[1:]
            key = store.create(command)
            print(json.dumps(store.launch(key), indent=2))
            return 0
        if action == "list":
            print(json.dumps(store.list(), indent=2))
            return 0
        if action == "status":
            print(json.dumps(store.get(args.job_id), indent=2))
            return 0
        if action == "logs":
            print(store.logs(args.job_id, getattr(args, "lines", 100)), end="")
            return 0
        if action == "retry":
            print(json.dumps(store.launch(args.job_id), indent=2))
            return 0
        if action == "_execute":
            return store.execute(args.job_id)
        raise ValueError(f"unknown job action {action}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Persistent detached WeightScript jobs"
    )
    parser.add_argument("--store", default=None)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("command", nargs=argparse.REMAINDER)
    sub.add_parser("list")
    for name in ("status", "retry", "_execute"):
        item = sub.add_parser(name)
        item.add_argument("job_id")
    logs = sub.add_parser("logs")
    logs.add_argument("job_id")
    logs.add_argument("--lines", type=int, default=100)
    args = parser.parse_args(argv)
    code = run_action(args)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
