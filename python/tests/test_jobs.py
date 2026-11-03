import os
import signal
import subprocess
import sys
import time

import pytest

from weightscript.jobs import JobStore, process_alive


def _wait_until(predicate, seconds=10):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_sigterm_is_forwarded_and_recorded(tmp_path):
    pidfile = tmp_path / "child.pid"
    script = (
        "import os,time,pathlib;"
        f"pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid()));"
        "time.sleep(30)"
    )
    store = tmp_path / "jobs"
    with JobStore(store) as jobs:
        key = jobs.create([sys.executable, "-c", script])
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "weightscript.jobs",
            "--store",
            str(store),
            "_execute",
            key,
        ]
    )
    try:
        assert _wait_until(pidfile.exists)
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
    child = int(pidfile.read_text())
    assert _wait_until(lambda: not process_alive(child))
    with JobStore(store) as jobs:
        job = jobs.get(key)
    assert job["status"] == "interrupted"
    assert "forwarded signal" in job["note"]
    assert not process_alive(child)


def test_killed_runner_is_diagnosed_and_not_retried_over_an_orphan(tmp_path):
    pidfile = tmp_path / "child.pid"
    marker = tmp_path / "marker"
    script = (
        "import os,sys,time,pathlib;"
        f"pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid()));"
        f"marker=pathlib.Path({str(marker)!r});"
        "sys.exit(0) if marker.exists() else time.sleep(30)"
    )
    store = tmp_path / "jobs"
    with JobStore(store) as jobs:
        key = jobs.create([sys.executable, "-c", script])
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "weightscript.jobs",
            "--store",
            str(store),
            "_execute",
            key,
        ]
    )
    child = None
    try:
        assert _wait_until(pidfile.exists)
        child = int(pidfile.read_text())
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)
        with JobStore(store) as jobs:
            job = jobs.get(key)
            assert job["status"] == "interrupted"
            assert job["diagnostics"]["child_alive"] is True
            with pytest.raises(ValueError, match="still running"):
                jobs.launch(key)
            with pytest.raises(ValueError, match="still running"):
                jobs.execute(key)
        os.kill(child, signal.SIGKILL)
        assert _wait_until(lambda: not process_alive(child))
        marker.write_text("go")
        with JobStore(store) as jobs:
            assert jobs.execute(key) == 0
            assert jobs.get(key)["status"] == "succeeded"
            assert jobs.get(key)["attempts"] == 2
    finally:
        if proc.poll() is None:
            proc.kill()
        if child and process_alive(child):
            os.kill(child, signal.SIGKILL)


def test_detached_tmux_job_survives_without_a_client(tmp_path):
    if not os.path.exists("/usr/bin/tmux"):
        pytest.skip("tmux is not installed")
    marker = tmp_path / "done.txt"
    script = (
        "import time,pathlib;"
        "time.sleep(1);"
        f"pathlib.Path({str(marker)!r}).write_text('ok')"
    )
    store = tmp_path / "jobs"
    session = None
    try:
        with JobStore(store) as jobs:
            key = jobs.create([sys.executable, "-c", script], cwd=tmp_path)
            launched = jobs.launch(key)
            session = launched["session"]
            saw_detached = False
            deadline = time.time() + 15
            while time.time() < deadline:
                job = jobs.get(key)
                if job["status"] == "running":
                    clients = subprocess.run(
                        ["tmux", "list-clients", "-t", session],
                        capture_output=True,
                        text=True,
                    )
                    assert clients.stdout.strip() == ""
                    saw_detached = True
                if job["status"] in {"succeeded", "failed", "interrupted"}:
                    break
                time.sleep(0.1)
            job = jobs.get(key)
        assert saw_detached
        assert job["status"] == "succeeded"
        assert marker.read_text() == "ok"
    finally:
        if session:
            subprocess.run(["tmux", "kill-session", "-t", session], check=False)


def test_hangup_of_attached_client_does_not_stop_the_job(tmp_path):
    import pty

    if not os.path.exists("/usr/bin/tmux"):
        pytest.skip("tmux is not installed")
    marker = tmp_path / "done.txt"
    script = (
        "import time,pathlib;"
        "time.sleep(1.5);"
        f"pathlib.Path({str(marker)!r}).write_text('ok')"
    )
    store = tmp_path / "jobs"
    session = None
    client = None
    try:
        with JobStore(store) as jobs:
            key = jobs.create([sys.executable, "-c", script], cwd=tmp_path)
            session = jobs.launch(key)["session"]
            assert _wait_until(
                lambda: jobs.get(key)["status"] in {"running", "succeeded"}
            )
            _master, slave = pty.openpty()
            client = subprocess.Popen(
                ["tmux", "attach", "-t", session],
                stdin=slave,
                stdout=slave,
                stderr=slave,
                start_new_session=True,
                env=dict(os.environ, TERM="xterm-256color"),
            )
            os.close(slave)
            time.sleep(0.3)
            if client.poll() is not None:
                pytest.skip(f"tmux attach exited early with status {client.returncode}")
            os.kill(client.pid, signal.SIGHUP)
            client.wait(timeout=5)
            deadline = time.time() + 15
            while time.time() < deadline:
                if jobs.get(key)["status"] in {"succeeded", "failed", "interrupted"}:
                    break
                time.sleep(0.1)
            job = jobs.get(key)
        assert job["status"] == "succeeded", job
        assert marker.read_text() == "ok"
    finally:
        if client and client.poll() is None:
            client.kill()
        if session:
            subprocess.run(["tmux", "kill-session", "-t", session], check=False)
