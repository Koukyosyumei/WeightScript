# WeightScript

Inspect a **trained Transformer**, edit a program describing its computation, and compile the edit back into a normal checkpoint—without gradient updates.

The practical backend supports **standard Hugging Face GPT-2 checkpoints**, including DistilGPT2. It preserves LayerNorm and softmax; it does not require a specially trained interpretable architecture. The current output is a numerical computation program with named circuits, not an automatically discovered short algorithm.

```text
trained checkpoint → computation IR + circuit traces
                   → agent/user edits IR
                   → same-architecture checkpoint + audit + undo package
                   → independent inference checks + held-out evaluation
```

## Install

Python 3.10+ and Rust are supported. Python performs tensor computation; the Rust CLI also exposes this backend.

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e '.[test]'
cargo build --release
export WEIGHTSCRIPT_PYTHON="$PWD/.venv/bin/python"
```

Use `weightscript-hf` from the activated environment, `.venv/bin/python -m weightscript.cli`, or `cargo run --release -- hf …`. For this session, the installed interpreter is `/dev/shm/weightscript-hf-env/bin/python`.

## Edit a real language model

```bash
weightscript-hf extract distilbert/distilgpt2 artifacts/program
weightscript-hf tensor distilbert/distilgpt2 transformer.h.0.attn.c_proj.weight --rows 0:8 --cols 0:8
```

The extraction contains `program.json`, checkpoint tensor names/shapes/statistics, the configuration, and the independent forward implementation for review. The program includes a source fingerprint and explicit bindings to Q/K/V, attention write projections, MLP read/write projections and LayerNorms.

Edit the `edits` array in `artifacts/program/program.json`, for example:

```json
[
  {"op": "head_scale", "layer": 0, "head": 2, "factor": 0.5},
  {"op": "neuron_scale", "layer": 3, "neurons": [12, 85], "factor": 0.0}
]
```

Then compile, generate and undo:

```bash
weightscript-hf compile distilbert/distilgpt2 artifacts/program/program.json artifacts/edited --tokenizer distilbert/distilgpt2
weightscript-hf generate artifacts/edited --text 'A computer program can' --max-new-tokens 32
weightscript-hf undo artifacts/edited artifacts/edited artifacts/restored
```

`undo` restores model tensors; tokenizer files remain available in the edited checkpoint directory. If generating from the restored checkpoint, pass `--tokenizer artifacts/edited`.

Supported edits:

| Program operation | Compiled weight change |
| --- | --- |
| `head_scale(layer, head, factor)` | Scale that head's rows in the attention output projection |
| `attention_scale(layer, head, factor)` | Scale that head's query weights and query bias |
| `neuron_scale(layer, neurons, factor)` | Scale selected MLP write rows |
| `remove_direction(layer, neurons, direction, strength)` | Subtract a rank-one residual-direction component from selected MLP write rows |

The direction is a nonzero list of `n_embd` numbers and is normalized by the compiler. For a selected write row `w`, the latter computes `w' = w - strength * dot(w, d̂) * d̂`. This expresses the family of targeted MLP corrections discussed in the weight-space-programming post. Direction discovery and neuron selection are supplied by an agent or researcher; they are not solved automatically.

An edit must target the exact checkpoint fingerprint. Shapes, configuration and weight tying are preserved. Output directories must be empty. The compiler saves regular `config.json` / `model.safetensors` files, an audit, and hash-checked original tensors for undo. No hooks or inference wrapper are needed to use the edited checkpoint.

## Inspect and verify

Use token IDs to avoid assumptions about custom research tokenizers:

```bash
weightscript-hf trace MODEL 2,0,1,0,0,3 artifacts/trace.json
weightscript-hf run MODEL artifacts/program/program.json 2,0,1,0,0,3
weightscript-hf verify MODEL artifacts/program/program.json inputs.jsonl artifacts/check.json
```

A trace includes attention matrices, per-head entropy, QK/OV singular values, and top activated neurons. `run` executes the independent computation program directly. `verify` compares both the unedited and edited programs against native Hugging Face inference, reporting absolute logit errors and tolerances. Input JSONL:

```json
{"input_ids": [2, 0, 1, 0, 3], "target": 0}
```

`target` is optional for verification; evaluation uses it as the expected **next token after the input**. Batches are grouped by sequence length without padding. These are numerical checks on supplied examples, **not universal formal proofs**.

## Run the trained-model research pilot

The upstream decompilation project supplies trained task checkpoints. Download only the small binary-majority checkpoint from a pinned revision:

```bash
python examples/fetch_research_checkpoint.py
python examples/research_benchmark.py artifacts/upstream-majority
```

This runs matched direct-tensor and program-edit controls, evaluates disjoint development and held-out length ranges, exports the selected checkpoint, and checks program/native and disk round-trip agreement. Results are saved under `artifacts/research/`. See [measured results](docs/results/pilot.md).

For an ordinary pretrained language model, run:

```bash
python examples/language_smoke.py distilbert/distilgpt2
```

This downloads about 330 MB, applies head scaling and sparse MLP direction removal, verifies inference, exports/reloads the model, and verifies undo. The edit tests the tooling; it is not claimed to improve the model.

## Compare editing approaches

```bash
weightscript-hf benchmark MODEL candidates.json dev.jsonl heldout.jsonl report.json --budget 50 --batch-size 32
```

`candidates.json` maps method names to proposed edits. The interfaces differ:

```json
{
  "direct_tensor": [
    {"raw": [{"op": "scale", "tensor": "transformer.h.0.attn.c_proj.weight", "rows": [0, 1], "factor": 0.5}]}
  ],
  "program_guided": [
    {"edits": [{"op": "neuron_scale", "layer": 0, "neurons": [0, 1], "factor": 0.5}]}
  ]
}
```

Those particular examples edit different circuits; construct matched interventions when testing compiler parity. Direct edits support matrix-row `scale` and `remove_direction`. Use `compile-raw MODEL raw.json OUTPUT` to export one with the same audit/undo support.

Candidate selection maximizes development next-token accuracy, breaking ties by target NLL. Each method receives the same maximum development forward-call budget. The selected candidate is evaluated once on held-out data. The report includes task accuracy/NLL, changes in all-position token predictions, reference KL, changed scalar counts, source/data hashes, software versions and evaluation call counts. Common base measurements and final/reference evaluations are reported separately from the selection budget. Candidate-generation costs, external agent calls and external inference are **not enforced or measured** by this harness. For a fair agent experiment, budget those separately and pre-register candidates before examining held-out results.

This is infrastructure for testing whether a program interface improves weight-space editing. It does **not yet beat or replicate** the linked frontier-LLM experiments. The pilot deliberately uses identical interventions in both interfaces; equal results are the expected control. See [research direction](docs/research.md) for the comparison still required.

## Scope and checks

- CPU float32, standard GPT2LMHeadModel, eager attention, causal inference without KV cache. DistilGPT2 uses this architecture. Cross-attention, upcast attention, pruned-head configurations, Qwen, Llama and custom architectures are currently unsupported.
- Arbitrary edits to Python forward source cannot be compiled. The supported edit surface is the typed JSON operations above.
- Exact computation extraction is separate from recovering a short symbolic algorithm. There is no D-RASP→Tracr conversion or trained-model algorithm simplifier here.
- The tool is agent-usable, but includes no autonomous LLM proposal policy or live external-model calls.
- Same architecture and unchanged tensors do not imply unchanged behavior. Held-out metrics are empirical and do not establish universal preservation.

```bash
pytest -q
cargo test
cargo clippy --all-targets -- -D warnings
```

`cargo run --release -- demo` also runs a small **compiled hard-attention** fixture with exact integer inference and exhaustive verification over 5,461 inputs. This is a compiler test fixture, not the trained-model research result.

Research references: [Decompiling Transformers](https://lacoco-lab.github.io/home/decompiling_transformers/), [Tracr](https://github.com/google-deepmind/tracr), [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). This is an independent implementation; none of those projects is vendored.
