"""Release contracts: lazy imports, CLI compatibility, public API and built artifacts."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import venv
import zipfile

import pytest


def run(*args, **kwargs):
    return subprocess.run(args, text=True, capture_output=True, **kwargs)


def test_help_and_version_do_not_import_model_libraries():
    # Block imports even on development machines where dependencies are installed.
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch','transformers','numpy','jax','tracr'}:
            raise AssertionError('heavy import: ' + fullname)
sys.meta_path.insert(0, Block())
import weightscript
assert weightscript.__version__ == '0.2.0'
assert 'extract' in dir(weightscript)
from weightscript.cli import main
try:
    main(sys.argv[1:])
except SystemExit as error:
    assert error.code == 0
"""
    for args in (
        ["--help"],
        ["--version"],
        ["extract", "--help"],
        ["rasp", "compile", "--help"],
        ["workbench", "--help"],
    ):
        result = run(sys.executable, "-c", script, *args)
        assert result.returncode == 0, result.stderr


def test_cli_validation_and_dependency_diagnostics(monkeypatch, capsys):
    from weightscript.cli import main
    from weightscript import cli

    for args in (["--threads", "0", "extract", "x", "y"], ["unknown"], ["extract"]):
        with pytest.raises(SystemExit) as error:
            main(args)
        assert error.value.code == 2
    original = sys.argv[:]

    def missing(args):
        raise ModuleNotFoundError("No module named 'z3'", name="z3")

    monkeypatch.setattr(cli, "execute", missing)
    assert main(["extract", "x", "y"]) == 3
    assert "weightscript[research]" in capsys.readouterr().err
    assert sys.argv == original


def test_expected_failures_and_interrupts(capsys):
    from weightscript._cli_support import run_cli

    for exception, code in (
        (ValueError("bad program"), 1),
        (OSError("missing file"), 1),
        (KeyboardInterrupt(), 130),
    ):

        def fail():
            raise exception

        assert run_cli(fail) == code
        assert "weightscript:" in capsys.readouterr().err
    with pytest.raises(RuntimeError):
        run_cli(lambda: (_ for _ in ()).throw(RuntimeError("bug")))


def test_public_api_roundtrip(tmp_path):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    import weightscript as ws

    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=11,
            n_positions=8,
            n_embd=8,
            n_layer=1,
            n_head=2,
            attn_pdrop=0,
            resid_pdrop=0,
            embd_pdrop=0,
            _attn_implementation="eager",
        )
    ).eval()
    before = ws.fingerprint(model)
    program = ws.extract(model)
    program["edits"] = [{"op": "head_scale", "layer": 0, "head": 0, "factor": 0.5}]
    edited = ws.compile_edits(model, program)
    assert ws.fingerprint(model) == before
    assert ws.audit(model, edited)["changed_scalars"] > 0
    ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        torch.testing.assert_close(
            ws.program_forward(model, program, ids),
            edited(ids).logits,
            atol=1e-6,
            rtol=1e-5,
        )
    ws.save_undo(model, edited, tmp_path)
    assert ws.fingerprint(ws.undo(edited, tmp_path)) == before
    with pytest.raises(ValueError):
        ws.undo(model, tmp_path)
    direct = ws.transaction(
        model,
        [
            {
                "op": "scale",
                "tensor": "transformer.h.0.attn.c_proj.weight",
                "select": [None, None],
                "value": 0.0,
            }
        ],
    )
    assert ws.audit(model, direct)["changed_scalars"] > 0


def test_sdist_wheel_installed_outside_checkout(tmp_path):
    pytest.importorskip("build")
    root = Path(__file__).resolve().parents[2]
    dist = tmp_path / "dist"
    result = run(
        sys.executable,
        "-m",
        "build",
        "--sdist",
        "--no-isolation",
        "--outdir",
        str(dist),
        str(root),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with tarfile.open(next(dist.glob("*.tar.gz"))) as archive:
        names = archive.getnames()
        assert any(name.endswith("/docs/api.md") for name in names)
        assert any(name.endswith("/python/tests/test_release.py") for name in names)
        archive.extractall(tmp_path / "source", filter="data")
    source = next((tmp_path / "source").iterdir())
    result = run(
        sys.executable,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(dist),
        str(source),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheel = next(dist.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        assert "weightscript/__main__.py" in archive.namelist()
        assert not any(name.startswith("tests/") for name in archive.namelist())
        metadata = archive.read("weightscript-0.2.0.dist-info/METADATA").decode()
        assert "Requires-Python: >=3.10" in metadata
    environment = tmp_path / "env"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "bin/python"
    # Reuse runtime dependencies without copying multi-GB torch installations.
    # A .pth path entry does not process the shared environment's editable hooks.
    site = run(
        str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"
    )
    assert site.returncode == 0, site.stderr
    dependency_paths = [
        p for p in sys.path if p.endswith(("site-packages", "dist-packages"))
    ]
    (Path(site.stdout.strip()) / "release-dependencies.pth").write_text(
        "\n".join(dependency_paths) + "\n"
    )
    env = dict(os.environ, OMP_NUM_THREADS="2", HF_HUB_OFFLINE="1")
    env.pop("PYTHONPATH", None)
    outside = tmp_path / "outside"
    outside.mkdir()
    result = run(
        str(python),
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-index",
        str(wheel),
        cwd=outside,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    script = """
from pathlib import Path
import sys, torch, weightscript as ws
from transformers import GPT2Config, GPT2LMHeadModel
assert Path(ws.__file__).is_relative_to(Path(sys.prefix))
torch.set_num_threads(2)
m = GPT2LMHeadModel(GPT2Config(vocab_size=11,n_positions=8,n_embd=8,n_layer=1,n_head=2)).eval()
m.save_pretrained('model')
"""
    result = run(str(python), "-c", script, cwd=outside, env=env)
    assert result.returncode == 0, result.stderr
    commands = [
        ["--version"],
        ["--help"],
        ["extract", "model", "program"],
        ["compile", "model", "program/program.json", "edited"],
        ["run", "model", "program/program.json", "1,2,3"],
        ["undo", "edited", "edited", "restored"],
        ["graph", "model", "graph.json"],
    ]
    for args in commands:
        result = run(str(environment / "bin/weightscript"), *args, cwd=outside, env=env)
        assert result.returncode == 0, result.stdout + result.stderr
    for executable, args in (
        (str(environment / "bin/weightscript-hf"), ["workbench", "--help"]),
        (str(python), ["-m", "weightscript.workbench", "--help"]),
        (str(python), ["-m", "weightscript", "--version"]),
    ):
        result = run(executable, *args, cwd=outside, env=env)
        assert result.returncode == 0, result.stderr
    assert json.loads((outside / "graph.json").read_text())
    # Execute existing numerical regression tests against the installed wheel.
    result = run(
        str(python),
        "-m",
        "pytest",
        str(source / "python/tests/test_core.py"),
        str(source / "python/tests/test_adapters.py"),
        "-q",
        cwd=outside,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_public_dependency_error_is_actionable():
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == 'torch':
            raise ModuleNotFoundError("No module named 'torch'", name='torch')
sys.meta_path.insert(0, Block())
import weightscript as ws
try:
    ws.load_model
except ws.MissingDependencyError as error:
    assert 'pip install weightscript' in str(error)
    assert isinstance(error, ImportError)
else:
    raise AssertionError('expected dependency error')
try:
    ws.nonexistent
except AttributeError:
    pass
else:
    raise AssertionError('expected attribute error')
"""
    result = run(sys.executable, "-c", script)
    assert result.returncode == 0, result.stderr
