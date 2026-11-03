# Python API

The supported public namespace is `weightscript`. Its exports resolve lazily;
plain import and `__version__` need no model dependencies. Accessing a model
function imports its implementation. Missing dependencies at that boundary raise
`weightscript.MissingDependencyError`, an `ImportError` subclass with an install
hint. Core validation retains `ValueError`; file operations retain `OSError`.
Implementation modules remain importable for compatibility but are experimental
unless listed here.

| Export | Contract |
| --- | --- |
| `load_model(source)` | Load a local path or HF ID through Transformers, CPU float32, eval mode; remote code disabled. |
| `extract(model)` | Return a JSON-compatible numerical program bound to model architecture and fingerprint. |
| `program_forward(model, program, input_ids, attention_mask=None, capture=False, validated=False)` | Execute a supported numerical adapter. Returns logits, or `(logits, traces)` when capturing. Inputs are rank-two integer tensors. Leave `validated=False` for normal use. |
| `compile_edits(model, program)` | Validate and apply supported program edits to a copied model; return the edited model. The source is preserved. |
| `transaction(model, edits)` | Apply a list of direct tensor operations atomically to a copied model, preserving tied storage and rejecting nonfinite results. |
| `audit(before, after)` | Return a JSON-compatible report of changed tensors/scalars and numerical differences. |
| `fingerprint(model)` | Return a SHA-256 binding to configuration and weights. |
| `save_undo(before, after, directory)` | Save original changed tensors and hashes in an existing directory. Returns `None`. |
| `undo(model, directory)` | Validate the edited fingerprint and undo package, return a restored copy, and verify the original fingerprint. |

Models for numerical adapters must be standard supported classes in eval mode
with CPU float32 parameters. GPT-2, Llama and Qwen2 adapters validate architecture
features; unsupported features raise `ValueError`. Generic direct transactions
can operate on other native models without implying semantic adapter support.
Programs are mutable dictionaries: edit their `edits` list, retain the extraction
metadata and fingerprint, and re-extract after changing the base model.

```python
from pathlib import Path
import torch
import weightscript as ws

model = ws.load_model("MODEL")
program = ws.extract(model)
program["edits"] = [{"op": "head_scale", "layer": 0, "head": 0, "factor": 0.5}]
with torch.inference_mode():
    logits = ws.program_forward(model, program, torch.tensor([[1, 2, 3]]))
edited = ws.compile_edits(model, program)
output = Path("artifacts/edited")
output.mkdir(parents=True, exist_ok=True)
edited.save_pretrained(output, safe_serialization=True)
ws.save_undo(model, edited, output)
restored = ws.undo(edited, output)
assert ws.fingerprint(restored) == ws.fingerprint(model)
```

`head_scale`, `attention_scale`, `neuron_scale` and `remove_direction` are
numerical edits, not recovered symbolic algorithms. Direct transactions use
`set`, `add`, `scale` or `low_rank_add`; see the [agent protocol](agent-protocol.md)
for proposal structures. Audits and undo prove recorded tensor changes and hash
restoration; they do not establish behavioral quality or universal equivalence.

The CLI `main(argv=None)` in `weightscript.cli` accepts an argument list without
modifying `sys.argv`, returns an integer status for runtime operations, and uses
argparse's `SystemExit` for help, version and syntax errors. Parser registration
and execution are separate in both CLI modules so additional orchestration
commands can integrate without loading models during help.
