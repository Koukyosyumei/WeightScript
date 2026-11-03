# WeightScript

WeightScript is a Python library and CLI for inspecting and editing trained
Transformer checkpoints without gradient updates. It exports a numerical program,
compiles supported edits into ordinary Hugging Face checkpoints, audits changed
weights, and saves hash-checked undo packages.

Requires Python 3.10+. CPU float32 is the tested execution path.

```bash
pip install weightscript
weightscript --help
weightscript --version
weightscript extract MODEL artifacts/program
```

For a source checkout, use `pip install .` (or `pip install -e '.[test]'` for
development). See [installation](docs/installation.md) for CPU PyTorch and optional
research dependencies. Help, version, and `import weightscript` do not import model
libraries or load checkpoints.

```python
import weightscript as ws

model = ws.load_model("MODEL")  # local checkpoint directory or Hugging Face ID
program = ws.extract(model)
program["edits"] = [{"op": "head_scale", "layer": 0, "head": 0, "factor": 0.5}]
edited = ws.compile_edits(model, program)
print(ws.audit(model, edited))
edited.save_pretrained("artifacts/edited", safe_serialization=True)
ws.save_undo(model, edited, "artifacts/edited")
```

Supported numerical adapters cover standard GPT-2, Llama and Qwen2 architectures.
Edits include head/attention scaling, neuron scaling and selected-neuron direction
removal. Generic tensor transactions support direct edits. Unsupported architecture
features are rejected by the semantic adapters. See the [public API](docs/api.md)
for contracts, errors, undo and verification.

```bash
weightscript compile MODEL artifacts/program/program.json artifacts/edited
weightscript verify MODEL artifacts/program/program.json inputs.jsonl report.json
weightscript undo artifacts/edited artifacts/edited artifacts/restored
weightscript graph MODEL graph.json
```

`inputs.jsonl` contains rows such as `{"input_ids":[1,2,3],"target":4}`.
`weightscript-hf` remains an alias, and `python -m weightscript` exposes the same
commands. The older `python -m weightscript.workbench` research interface and
`weightscript-hf workbench ...` continue to work.

Optional research workflows include upstream component pruning, Tracr RASP
compilation, bounded rule synthesis and trusted external editing agents. Numerical
expansion is not algorithm discovery; behavioral hypotheses and opaque computations
are recorded explicitly. Verification measures supplied inputs or a declared finite
domain, not universal equivalence. No claim of superiority to prior research is made.
See [research status](docs/research.md), [recorded results](docs/results/framework.md)
and the [agent protocol](docs/agent-protocol.md).

The separate Rust executable is an exact integer compiler fixture (`cargo run --
demo`), not the trained-model runtime. Its legacy `hf` forwarding command uses the
installed Python package selected by `WEIGHTSCRIPT_PYTHON` (default `python3`). It
never injects a build checkout into Python's import path. Prefer the Python CLI for
trained-model work.

Licensed under Apache-2.0.
