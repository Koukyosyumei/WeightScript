# Installation

Use Python 3.10 or newer in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .
weightscript --help
```

WeightScript is not published on PyPI. Install from this source checkout, or from
a wheel built locally with `python -m build`. Do not use `pip install weightscript`
unless a future release is actually published.

For CPU-only systems, install the desired PyTorch CPU wheel first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install .
```

Development and release tools are available with `pip install -e '.[test,build]'`.
Build with `python -m build` and run `pytest`. A wheel contains only the Python
package; the sdist also includes documentation, Python tests and the independent
Rust fixture sources.

The runtime dependencies are PyTorch, Transformers (pinned for adapter
compatibility), safetensors and NumPy. Native edits do not require research tools.
No model is downloaded by help, version or package import. Loading a Hugging Face
ID may download a model; use a local directory and `HF_HUB_OFFLINE=1` for offline
operation with existing checkpoints. Model execution uses CPU float32 and can
require substantial RAM.

## Research workflows

```bash
pip install -e '.[research]'
pip install 'tracr @ git+https://github.com/google-deepmind/tracr@9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd'
git clone https://github.com/lacoco-lab/decompiling_transformers upstream-decompiler
git -C upstream-decompiler checkout 787398b0f4393a90ad02cd8e71be48d1e85d2b21
weightscript decompile MODEL calibration.jsonl artifacts/decompiled \
  --checkout upstream-decompiler --evaluation evaluation.jsonl
```

The research extra includes JAX/Haiku, constraint solvers, and upstream plotting
and graph dependencies. Tracr and the upstream decompiler are separate, explicit
installations; neither is vendored. A missing research module reports an install
hint and status 3. External agent commands also require their own executable and,
for an external TUI, tmux and that provider's own authenticated CLI. `agent-start
--provider grok` discovers the installed `grok` executable from its help text.
It does not log in or invent an agent reply. Codex remains available through
`codex-start` and is not started unless that command is used.

## Compatibility and exit status

`weightscript`, `weightscript-hf`, `python -m weightscript` and
`python -m weightscript.cli` share the unified command surface. Existing workbench
commands are now available directly alongside extract/compile/verify. The older
`python -m weightscript.workbench` interface remains supported. Global `--threads`
(default 2) precedes the command and must be positive.

Status 0 means success, 1 means an operation or verification failed, 2 means an
argument error, 3 means a missing Python dependency, and 130 means interruption.
Unexpected programming errors retain their traceback. A bounded rewrite that
fails its oracle does not export a checkpoint.

The optional Rust binary has the same executable name as the Python CLI; avoid
placing it earlier on PATH when using trained-model commands. `cargo run -- hf
--help` forwards through the installed Python package. Set `WEIGHTSCRIPT_PYTHON`
to the interpreter in your chosen virtual environment when necessary.
