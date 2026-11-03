# Research status and acceptance criteria

The engineering framework implements both editing paths, upstream integrations, a bounded trained-rule round trip, generic transactions, architecture adapters and a budgeted agent controller. That does not complete the research objective of reliably improving arbitrary LLMs or outperforming the reference work.

| Roadmap item | Evidence / current scope | Remaining acceptance criterion |
| --- | --- | --- |
| Run upstream decompiler | Pinned engines for component, full path, and QK run locally on trained majority, unique copy, and sort. Majority component mask 13 to 3 edges with full-span replay 512/512; later stages replay 386/512 and are rejected. Copy/sort later stages miss most answer tokens. No QK library primitive accepted. See `docs/decompilation.md` | Paper coefficient search, MLP primitive replacement, and faithful full-sequence stage equivalence |
| Run Tracr | Real pinned compiler; JSON frontend; independent serialized-weight inference; bounded verification | Expand supported primitive families without assuming every RASP program compiles faithfully |
| Common representation | Versioned nodes, weight references, provenance, fidelity, opaque/numerical/hypothesis/primitive status | More detailed activation/path provenance across additional backends |
| Program-to-existing-weight editing | Output-span LP lowers count, position, min selection, and span-next rules. Trained majority invert changes 48 scalars and scores 510/512 on held-out length 9. Four-prefix copy-to-sort and sort-to-copy patches score 0/4 held out; the sort patch stops on the variable budget | Sequence-wide edits of copy and sort, and edits whose held-out accuracy stays complete |
| Direct editing | Arbitrary tensor selections, scalar/tensor updates, low-rank additions; atomic source preservation and undo | Large-model memory/device optimizations |
| Agent execution | External JSON policies; Codex and Grok TUI bridges; durable sessions. One live Grok program-mode seed on trained majority evaluated one count rule, rejected it on development NLL, and kept the original weights. Verbatim cost was 0.02851104 USD across two turns. See `docs/results/grok-majority-pilot.json` | Repeated strong-agent seeds and independent public task suites beyond this single rejected proposal |
| Common evaluation | Disjoint domains, hashes, accuracy/NLL/uncertainty, generation/loop metrics, collateral KL/disagreement, usage/cost records | Task-specific grading and public benchmark reproduction at the reference scale |
| Architectures | GPT-2, Llama, Qwen2, and Mistral numerical adapters. RoPE default, linear, dynamic NTK, and Llama-3 scaling. Qwen2 and Mistral sliding windows. See `docs/model-support.md` | Qwen3.5 and other hybrid decoders, YaRN, LongRoPE, and families without an explicit contract |
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

## Public matched-arm measurement

`examples/public_comparison.py` scored three seeds and four development proposals per selective arm. Held-out and collateral rows were scored after selection. The upstream checkout was revision `787398b0f4393a90ad02cd8e71be48d1e85d2b21`. Checkpoint names are `bin_majority-@1l1h16d3lr01drop`, `unique_copy-@2l1h64d3lr01drop`, and `sort-@1l1h256d3lr01drop`. The language rows are 16-token windows from the public-domain opening of Pride and Prejudice (passage sha256 `59b9b35994bdc1df809fdf02be72d208ba7fe0f5c865089f5a78031f0ef45d89`): 16 development windows, 16 held-out windows, and 8 collateral windows from the next chapter. Those language windows are repeated across seeds. The restored language fingerprint is `b424a1ed3130606acdc0cc58f02e7df4cbc6613dcaaa8730ba0e822307fe76c7` (81,912,576 parameters, GPT-2 family), taken from the undo package stored with the smoke edit. Exact rows are in `docs/results/public-baselines.json`.

`methods_better_than_untouched_on_every_seed` is empty for every task. A method is listed only when held-out accuracy is strictly higher and held-out NLL is strictly lower on every seed.

Majority held-out accuracy is 1.0 on all 16 examples for every arm and seed. Untouched held-out NLL is 0.0214, 0.0270, and 0.0236. The shared head-scale arms (scaling, direct, and program) lower it to 0.0052, 0.0053, and 0.0061. Direction lowers it to 0.0056, 0.0056, and 0.0065. Accuracy does not move. These are confidence changes on an already correct set, with Wilson intervals that still reach 1.0.

Unique copy held-out accuracy is 1.0. Untouched held-out NLL is about 0.0002 to 0.0003. The head-scale arms raise it to about 0.0008 to 0.0013.

Sort does not produce a cross-seed winner. On seed 0, direction raises development accuracy from 0.970 to 0.985 and lowers held-out accuracy from 1.000 to 0.986. On seed 2, the head-scale arms keep held-out accuracy at 0.986 and raise held-out NLL from 0.0822 to 0.1351.

On the language windows every arm keeps development accuracy 0.188 and held-out accuracy 0.438. Direction changes 2,359,213 scalars and lowers held-out NLL from 3.5982 to 3.5695, with collateral KL about 0.00257. The three seeds repeat the same windows and the same deterministic edit. That is one 16-example fluctuation, not a language-quality result, and it is separate from the symbolic count-rule result of 2,047/2,048.

Direct, program, and scaling match on these runs because they apply one head-scale grid. That match is an interface control.

## Live Grok editing pilot

`examples/grok_edit_pilot.py` ran one program-mode session on `bin_majority-@1l1h16d3lr01drop`, seed 0, with the same development, held-out, and collateral splits as the matched-arm runner. Held-out rows were not sent to the policy. The policy was the installed Grok CLI, one turn at a time, and the usage fields below are copied from `grok --output-format json`.

The first turn evaluated `count_threshold` at 0.5 with `invert` false. Development accuracy stayed 1.0 and development NLL moved from 0.01742 to 0.01882. The patch changed 16 scalars in the final LayerNorm bias and was not selected. The second turn finished. The retained checkpoint is the original model: 0 changed scalars, held-out accuracy 1.0, held-out NLL 0.02138, collateral accuracy 1.0. Selection forwards were 8 of 16. Wall time was 77.6 seconds.

Turn costs were 0.01730668 USD (9,548 input tokens, 4,565 output tokens, 22,945 total) and 0.01120436 USD (10,045 input tokens, 1,408 output tokens, 20,285 total). The model id in the wrapper was `grok-4.7-build`. This is one seed and one rejected rule, not an agent-quality ranking.

## What remains open

Paper coefficient search, MLP primitive replacement, faithful full-sequence stage equivalence, sequence-wide copy and sort edits, and a statistically supported improvement over matched strong agents are open. Qwen3.5 and other hybrid decoders stay unsupported. Unknown QK paths stay opaque. Do not use final evaluation failures for refinement.

The motivating weight-space post's Qwen math result is not directly comparable to a binary-rule rewrite. Reproducing that experiment also requires the corresponding architecture, generation settings, task data, grader and a capable editing agent. We must not report a tool-interface pilot as a defeat of that result.

## References

- [Decompiling Transformers](https://lacoco-lab.github.io/home/decompiling_transformers/)
- [Tracr](https://github.com/google-deepmind/tracr)
- [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
