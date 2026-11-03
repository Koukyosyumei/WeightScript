# Research status and acceptance criteria

The engineering framework implements both editing paths, upstream integrations, a bounded trained-rule round trip, generic transactions, architecture adapters and a budgeted agent controller. That does not complete the research objective of reliably improving arbitrary LLMs or outperforming the reference work.

| Roadmap item | Evidence / current scope | Remaining acceptance criterion |
| --- | --- | --- |
| Run upstream decompiler | Pinned engines for component, full path, and QK run locally on trained majority, unique copy, and sort. Majority component mask 13 to 3 edges with full-span replay 512/512; later stages replay 386/512 and are rejected. Copy/sort later stages miss most answer tokens. No QK library primitive accepted. See `docs/decompilation.md` | Paper coefficient search, MLP primitive replacement, and faithful full-sequence stage equivalence |
| Run Tracr | Real pinned compiler; JSON frontend; independent serialized-weight inference; bounded verification | Expand supported primitive families without assuming every RASP program compiles faithfully |
| Common representation | Versioned nodes, weight references, provenance, fidelity, opaque/numerical/hypothesis/primitive status | More detailed activation/path provenance across additional backends |
| Program-to-existing-weight editing | Output-span LP lowers count, position, min selection, and span-next rules. Trained majority invert changes 48 scalars and scores 510/512 on held-out length 9. Four-prefix copy-to-sort and sort-to-copy patches score 0/4 held out; the sort patch stops on the variable budget | Sequence-wide edits of copy and sort, and edits whose held-out accuracy stays complete |
| Direct editing | Arbitrary tensor selections, scalar/tensor updates, low-rank additions; atomic source preservation and undo | Large-model memory/device optimizations |
| Agent execution | External JSON policies; separate authenticated Codex TUIs running direct/program/hybrid pilots; local policy option; step/time/inference allowances, errors, journals | Repeated strong-agent seeds and independent public task suites beyond the exploratory 16-probe pilot |
| Common evaluation | Disjoint domains, hashes, accuracy/NLL/uncertainty, generation/loop metrics, collateral KL/disagreement, usage/cost records | Task-specific grading and public benchmark reproduction at the reference scale |
| Architectures | GPT-2, Llama, Qwen2 numerical adapters; actual DistilGPT2, SmolLM2-135M and Qwen2.5-0.5B-Instruct checks; GQA tests; opaque native fallback | Scaled RoPE, sliding windows, Qwen3.5 hybrid architecture and other model families |
| Hybrid editing | Compose symbolic rule lowering, numerical edits and tensor transactions; numerical explanation of direct differences | Demonstrate that hybrid proposal quality exceeds both separate arms |
| Surpass prior methods | Executable comparisons and retained unsuccessful runs | **Not achieved**: statistically supported improvement over matched strong agents and reference methods |

## What has actually been established

- On the trained binary-majority checkpoint, component pruning keeps uniform token aggregation into an opaque MLP. Full-span replay of that component stage matches the native model on 512/512 length-9 answers. The full-path and QK engines match 386/512 and are rejected. Logit error stays near 5, so the claim is decision agreement under calibrated linear LayerNorm.
- The same driver on unique copy and sort preserves the first answer token more often than the rest of the answer. Direct QK weight matching leaves the remaining attention paths opaque. Details, domains, and counterexamples are in `docs/decompilation.md`.
- A finite behavioral library matches majority and copy on held-out strings. Sort agrees with the model on 92/96 calibration positions, below the 0.99 bar, so the native model stays in place. This is activation/behavior-informed synthesis, not weight-only discovery or a universal decompilation result.
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
