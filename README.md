# WeightScript

A framework for **program-guided and direct editing of trained Transformer weights**. Both paths produce ordinary, auditable checkpoints and share evaluation, provenance and undo facilities.

```text
trained weights ── extraction / circuit discovery ── program ── edit ── lowering ─┐
       └────────────────── direct tensor transaction ──────────────────────────┤
                                                           verify / evaluate ─┴─ edited model
```

The framework now executes the real **decompiling_transformers** component-pruning implementation and the real **Tracr** compiler. It also provides independent GPT-2, Llama and Qwen2 execution adapters, bounded counterexample-guided rule compilation into a trained GPT-2, and an external-agent protocol.

Interpretation status is explicit: **opaque computation**, **numerical operation**, **behavioral hypothesis**, or **known primitive**. Numerical forward expansion alone is not called algorithm discovery. Unknown computations are retained, rather than silently replaced with guessed algorithms.

## Installation

Python 3.10+; Python 3.12 is used in CI. CPU float32 is the tested execution path. Rust provides a CLI launcher and an exact integer compiler fixture; the trained-model framework uses Python/PyTorch.

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e '.[test,research]'
.venv/bin/pip install 'tracr @ git+https://github.com/google-deepmind/tracr@9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd'
source .venv/bin/activate
```

The research extra installs JAX/Haiku, plotting dependencies required by upstream imports, and the constraint solvers. The base package can run native model editing without those dependencies. No upstream project is vendored.

In this workspace, `/dev/shm/weightscript-hf-env/bin/python` has the tested dependencies. All commands can be run as `python -m weightscript.workbench` or `python -m weightscript.cli` without installing entry points. For Rust:

```bash
export WEIGHTSCRIPT_PYTHON=/dev/shm/weightscript-hf-env/bin/python
cargo run --release -- hf workbench --help
```

## Actual upstream decompilation

```bash
git clone https://github.com/lacoco-lab/decompiling_transformers /dev/shm/weightscript-decompiler
git -C /dev/shm/weightscript-decompiler checkout 787398b0f4393a90ad02cd8e71be48d1e85d2b21
python examples/fetch_research_checkpoint.py
python examples/research_benchmark.py artifacts/upstream-majority
weightscript decompile artifacts/upstream-majority artifacts/research/dev.jsonl artifacts/decompiled \
  --checkout /dev/shm/weightscript-decompiler \
  --evaluation artifacts/research/heldout.jsonl
```

This invokes upstream masks, activation/path hooks, configuration pruning and source generation. Our deterministic greedy zero-ablation driver replaces the paper's optimization-based mask search; this is **component-stage integration, not a reproduction of its entire three-stage pipeline**. Original checkpoint weights remain untouched.

Saved path programs can be executed with `weightscript run-decompiled MODEL artifact.json '[2,0,1,0,3]' --checkout CHECKOUT`.

Native LayerNorm is the default. `--linear-ln` enables calibration-domain variance linearization, with its fidelity measured separately. The artifact records the upstream revision, checkpoint hash, retained paths, ablation trials and calibration/evaluation agreement. A machine-local `my_paths` constant missing from the upstream checkout is supplied at import time; its algorithm source is not modified.

On the trained binary-majority checkpoint, this produced:

```text
a0-0 = Attention(Q=[], K=[], V=token)
m0 = MLP(inp=a0-0)
output = Unembed(inp=m0)
```

The remaining MLP is still opaque until an independently evaluated primitive hypothesis is attached.

## Trained weights → discovered rule → edited weights

```bash
python examples/trained_rule_roundtrip.py artifacts/upstream-majority \
  --checkout /dev/shm/weightscript-decompiler --output artifacts/rule-roundtrip
```

This executes both lowering routes:

1. Upstream causal pruning → finite-library behavioral primitive synthesis → a binary count-threshold hypothesis. This uses model inference and is **not weight-only decompilation**.
2. Edit the hypothesis from majority to minority.
3. **New model:** lower the program through real Tracr, then execute its serialized weights with an independent NumPy Transformer runtime.
4. **Existing model:** solve a bounded rank-one MLP write patch with an LP, check its recorded inequalities with Z3, and test native float32 behavior. Counterexamples refine the synthesis domain. Final held-out inputs never enter the refinement loop.
5. Export/reload and verify undo restores the original checkpoint hash.

The existing-model compiler changes the final MLP write weights/bias and may zero the final LayerNorm bias to make decision constraints linear. Every changed tensor is audited. It observes hidden activations on a specified domain and solves constraints; it is **activation-informed weight synthesis**, not zero-data editing or an absence of optimization. There are no gradient updates.

Recorded result: 48 changed scalars, two refinement rounds, 2,047/2,048 correct on untouched length-11 inputs for the new minority task. The remaining failure is preserved in the evidence. Original majority behavior is intentionally changed; all-position collateral changes are large. This is a controlled rule-editing result, not a claim of improved general language capability.

Use the stages individually:

```bash
weightscript synthesize MODEL calibration.jsonl artifacts/hypothesis \
  --evaluation evaluation.jsonl --causal-artifact artifacts/decompiled/artifact.json
weightscript rasp compile artifacts/hypothesis/program.rasp.json artifacts/tracr
weightscript rasp run artifacts/tracr '[2, 0, 1, 0, 3]'
weightscript rasp verify artifacts/tracr artifacts/hypothesis/program.rasp.json --max-len 4
weightscript rewrite MODEL rule.json synthesis.jsonl oracle.jsonl artifacts/rewrite --heldout final.jsonl
```

`rule.json`: `{"op":"count_threshold","threshold":0.5,"invert":true}`. The current symbolic synthesis/constraint compiler supports binary count rules; other behavior remains opaque or requires an additional lowering backend.

## RASP frontend and same-layout Tracr patching

The JSON frontend represents RASP tokens/indices, maps, sequence maps, selectors, aggregation and selector width. Expressions use a small allowlist; arbitrary Python source is not executed. See `previous_program()` and `count_program()` in `python/weightscript/symbolic.py` for executable examples.

```bash
weightscript rasp compile before.rasp.json artifacts/tracr-before
weightscript rasp patch artifacts/tracr-before after.rasp.json artifacts/tracr-after
weightscript rasp recover artifacts/tracr-before recovered.json --library candidate-programs.json
```

`patch` refuses changes in architecture, residual basis, encoders or shapes. The previous-token trigger edit changes six scalars and passed exhaustive comparison on 340 inputs. `recover` recognizes exact weight identity in a supplied finite candidate library; it is not a general Tracr decompiler and never consults stored source metadata.

Serialized models use safetensors plus a JSON manifest; inference does not recompile the source program. Compiler smoke checks fail closed on mismatches. For example, upstream numerical aggregate basis inference does not cover arbitrary binary fractions; count lowering uses selector widths instead. For causal count programs, the declared observation is the last position; a noncausal RASP interpreter is not assumed equivalent at intermediate prefix positions.

## Edit ordinary language-model checkpoints

```bash
weightscript-hf extract HuggingFaceTB/SmolLM2-135M artifacts/program
weightscript-hf tensor HuggingFaceTB/SmolLM2-135M model.layers.0.self_attn.o_proj.weight --rows 0:8 --cols 0:8
```

Edit the `edits` array in `program.json`:

```json
[
  {"op":"head_scale","layer":0,"head":2,"factor":0.5},
  {"op":"neuron_scale","layer":3,"neurons":[12,85],"factor":0.0}
]
```

Supported numerical edits: `head_scale`, `attention_scale`, `neuron_scale`, and selected-neuron `remove_direction`. The latter performs `w' = w - strength * dot(w, d̂) * d̂` on write vectors, respecting each architecture's matrix orientation and grouped-query layout.

```bash
weightscript-hf compile MODEL artifacts/program/program.json artifacts/edited --tokenizer MODEL
weightscript-hf verify MODEL artifacts/program/program.json inputs.jsonl artifacts/check.json
weightscript-hf generate artifacts/edited --text 'A computer program can' --max-new-tokens 32
weightscript-hf undo artifacts/edited artifacts/edited artifacts/restored
```

The compiler preserves shapes, configuration and weight tying. It binds edits to the source fingerprint, saves ordinary Hugging Face checkpoints and a hash-checked undo package. Verification compares independent program execution with native model execution. It is a numerical check on supplied inputs, not a universal formal theorem.

Tested real checkpoints: **DistilGPT2, 81.9M parameters**, **SmolLM2-135M, 134.5M parameters**, and **Qwen2.5-0.5B-Instruct, 494M parameters**. SmolLM2 exercises RMSNorm, RoPE and gated MLPs. Qwen2/GQA has independent adapter tests and a real 494M-parameter checkpoint smoke test. Scaled/dynamic RoPE, sliding-window attention, Qwen3.5 hybrid layers and custom subclasses require additional adapters. Native opaque graphs and generic direct transactions are available without a semantic adapter, subject to HF native-model loading and available memory.

## Generic direct editing and explaining changes

```json
{
  "transaction":[
    {"op":"scale","tensor":"transformer.h.0.attn.c_proj.weight","select":[[0,1],null],"value":0.5}
  ]
}
```

```bash
weightscript-hf compile-raw MODEL transaction.json artifacts/direct
weightscript explain MODEL artifacts/direct artifacts/explanation.json
weightscript graph MODEL artifacts/graph.json
```

Transactions support `set`, `add`, `scale` over arbitrary tensor dimensions and `low_rank_add` with explicit factors. Axis selectors can be full axes, index lists or bounded slices. They are atomic with respect to the source model, preserve tied storage, and reject nonfinite results. The legacy matrix-row `raw` interface remains supported.

`explain` maps a direct edit back into numerical operations such as projection scaling and rank-one write updates, retaining unrecognized changes as opaque. Numerical low rank alone is not presented as a semantic explanation.

## Autonomous editing sessions

Use a **separate Codex TUI as editing agent** and keep the edited model local:

```bash
weightscript codex-start /dev/shm/weightscript-agent --session weightscript-agent
weightscript session MODEL dev.jsonl artifacts/codex-session \
  --policy 'python -m weightscript.tmux_policy --workspace /dev/shm/weightscript-agent --session weightscript-agent' \
  --goal 'Desired improvement' --mode hybrid --heldout heldout.jsonl --policy-timeout 260
```

This inherits existing Codex authentication and configured model, starts a standalone TUI to keep experiment arms independent, and uses files for controller requests and genuine Codex responses. Inspect with `tmux attach -t weightscript-agent`. No new API key is required. For a reproducible small-target pilot, see `examples/codex_tui_study.py`.

A small local model can alternatively serve as a policy backend for protocol testing:

```bash
weightscript session MODEL dev.jsonl artifacts/session \
  --policy 'python examples/local_policy.py --model Qwen/Qwen2.5-0.5B-Instruct' \
  --goal 'Describe the desired behavior change' --mode hybrid \
  --heldout heldout.jsonl --collateral collateral.jsonl \
  --max-forwards 50 --max-steps 10 --policy-timeout 180
```

Modes: `direct`, `program`, `hybrid`. A trusted external policy receives JSON on stdin and emits one JSON action on stdout. Tools include bounded weight inspection, circuit tracing, generation, candidate evaluation and finish. Hybrid proposals can compose rule lowering, numerical program edits and direct transactions.

The controller records the protocol before proposals, retains errors, enforces step/time limits and a conservative inference-call reservation budget, and selects candidates using development accuracy/NLL. Invalid or interrupted operations can consume reserved allowance. Generation reserves its maximum token count, then refunds unused calls. Hidden probes made by the built-in constraint compiler are charged. External policy-model token/call usage is reported separately when supplied by the policy. Final and collateral evaluation are outside selection budgets and never sent to the policy.

**This is not a security sandbox.** Policy commands are trusted processes with host access; inference outside the framework cannot be enforced or independently metered. No API key is required for the included local Qwen policy, but it needs a model download and CPU time. A small local model can fail to use tools or produce useful edits; such failures are results, not silently repaired proposals.

```bash
python examples/local_agent_study.py artifacts/upstream-majority
```

This is an exploratory two-arm local-agent case study, not a reproduction of the frontier-agent blog. It records invalid actions as well as scores. A serious superiority claim requires multiple agents, tasks, seeds, comparable proposal-generation costs and independent evaluation.

## Evaluation and checks

JSONL next-token data: `{"input_ids":[2,0,1,0,3],"target":0}`. Generation data optionally adds `expected_ids`. Next-token metrics include accuracy, Wilson intervals, target NLL and its standard error, reference KL and token disagreement. Generation metrics include exact token match, token-limit rate and repeated four-grams.

```bash
weightscript-hf benchmark MODEL candidates.json dev.jsonl heldout.jsonl report.json --budget 50
weightscript evaluate-generation MODEL generation.jsonl generation-report.json --max-new-tokens 64
python examples/language_smoke.py HuggingFaceTB/SmolLM2-135M --output artifacts/smoke
WEIGHTSCRIPT_UPSTREAM=/dev/shm/weightscript-decompiler pytest -q
ruff check --select E9,F python examples
ruff format --check python examples
cargo test
cargo clippy --all-targets -- -D warnings
```

The legacy integer Rust fixture (`cargo run --release -- demo`) remains a bounded exhaustive compiler test, separate from the trained-model evidence.

See [current results](docs/results/framework.md), [research status and remaining acceptance criteria](docs/research.md), and [agent protocol](docs/agent-protocol.md).

References: [Decompiling Transformers](https://lacoco-lab.github.io/home/decompiling_transformers/), [Tracr](https://github.com/google-deepmind/tracr), [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). WeightScript does not yet demonstrate superiority to those research results.
