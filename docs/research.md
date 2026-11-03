# Research status and acceptance criteria

The engineering framework implements both editing paths, upstream integrations, a bounded trained-rule round trip, generic transactions, architecture adapters and a budgeted agent controller. That does not complete the research objective of reliably improving arbitrary LLMs or outperforming the reference work.

| Roadmap item | Evidence / current scope | Remaining acceptance criterion |
| --- | --- | --- |
| Run upstream decompiler | Real upstream component hooks/masks/source renderer; pinned revision; learned checkpoint pruned from 13 paths to 3 | Full paper three-stage pruning/primitive pipeline and broader task reproduction |
| Run Tracr | Real pinned compiler; JSON frontend; independent serialized-weight inference; bounded verification | Expand supported primitive families without assuming every RASP program compiles faithfully |
| Common representation | Versioned nodes, weight references, provenance, fidelity, opaque/numerical/hypothesis/primitive status | More detailed activation/path provenance across additional backends |
| Program-to-existing-weight editing | Six-element same-layout Tracr rule patch; 48-scalar trained GPT-2 count-rule patch using bounded constraints | Generic architecture-preserving semantic lowering beyond the binary count family |
| Direct editing | Arbitrary tensor selections, scalar/tensor updates, low-rank additions; atomic source preservation and undo | Large-model memory/device optimizations |
| Agent execution | External JSON policies; separate authenticated Codex TUIs running direct/program/hybrid pilots; local policy option; step/time/inference allowances, errors, journals | Repeated strong-agent seeds and independent public task suites beyond the exploratory 16-probe pilot |
| Common evaluation | Disjoint domains, hashes, accuracy/NLL/uncertainty, generation/loop metrics, collateral KL/disagreement, usage/cost records | Task-specific grading and public benchmark reproduction at the reference scale |
| Architectures | GPT-2, Llama, Qwen2 numerical adapters; actual DistilGPT2, SmolLM2-135M and Qwen2.5-0.5B-Instruct checks; GQA tests; opaque native fallback | Scaled RoPE, sliding windows, Qwen3.5 hybrid architecture and other model families |
| Hybrid editing | Compose symbolic rule lowering, numerical edits and tensor transactions; numerical explanation of direct differences | Demonstrate that hybrid proposal quality exceeds both separate arms |
| Surpass prior methods | Executable comparisons and retained unsuccessful runs | **Not achieved**: statistically supported improvement over matched strong agents and reference methods |

## What has actually been established

- Upstream path pruning recovers uniform token aggregation feeding an opaque MLP on the trained binary-majority checkpoint, with measured final-token fidelity.
- A finite behavioral library matches a count-threshold hypothesis. This is activation/behavior-informed synthesis, not weight-only discovery or a universal decompilation result.
- Editing that rule can generate a new model with Tracr, or patch the original model by solving a restricted rank-one write problem.
- The constrained compiler needs explicit activation probes. It uses an LP rather than gradient descent. Auxiliary rational inequality checking is not an IEEE float32 proof. Native execution is tested separately.
- Counterexamples matter: the initial patch fails on two length-9 inputs. Two refinement rounds satisfy that bounded oracle. One length-11 input still fails; final held-out accuracy is 2,047/2,048. These failures remain in the report.
- A real 134.5M-parameter language model supports numerical program editing, export/reload and undo. The smoke-test edit is not claimed to improve language quality.

## Next research experiment

Fix an independent editing-agent backend, at least three trained model/task families, separate target and collateral datasets, and multiple seeds. Match policy context, model/tool inference, wall-time and proposal counts. Include untouched, random-edit, direct raw-weight, simple scaling, global direction-removal, program and hybrid arms. Charge compiler activation probes and report solver time. Do not use final evaluation failures for refinement.

The motivating weight-space post's Qwen math result is not directly comparable to a binary-rule rewrite. Reproducing that experiment also requires the corresponding architecture, generation settings, task data, grader and a capable editing agent. We must not report a tool-interface pilot as a defeat of that result.

## References

- [Decompiling Transformers](https://lacoco-lab.github.io/home/decompiling_transformers/)
- [Tracr](https://github.com/google-deepmind/tracr)
- [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
