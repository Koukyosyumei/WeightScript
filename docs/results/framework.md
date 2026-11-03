# Framework integration and experiment evidence

These results distinguish working engineering paths, controlled task edits, unsuccessful agent experiments and unproven research claims.

## Real upstream component extraction

The pinned `lacoco-lab/decompiling_transformers` implementation ran on its own gradient-trained binary-majority checkpoint (5,872 parameters). Greedy zero-ablation, using upstream masks and hooks, reduced 13 component paths to 3. The upstream source renderer produced uniform token aggregation → MLP → unembedding. Final-token agreement on separately generated evaluation examples was 100%; logits differed. The full paper's optimized three-stage pipeline is not reproduced by this driver.

Extracted artifacts now include executable edge masks and normalization constants. `weightscript run-decompiled` can replay them against the fingerprinted source checkpoint without rerunning pruning. Opaque MLPs remain opaque unless separately interpreted.

## Real Tracr same-layout rule patch

The program originally copies the previous token when the current token is 0. The edited condition accepts 0 or 1. Tracr compiled both; the residual layout was checked before patching. Six scalars changed, in a single MLP read matrix. Architecture and encoders were preserved. Original and edited models each passed exhaustive output comparison on all 340 sequences over four symbols of lengths 1–4.

Recovery recognizes exact tensor identity in a provided finite program library. This is deliberately restricted and independent of embedded source metadata. Serialized inference uses the saved matrices and a separate NumPy Transformer implementation.

[Report](tracr-roundtrip.json).

## Rule extraction and editing a gradient-trained model

Causal component evidence was combined with behavioral matching over a declared library: count thresholds, parity, first/last bit and constants. A majority count rule matched all 170 calibration examples and 512 evaluation examples. The hypothesis was edited to minority.

| Route | Result |
| --- | --- |
| Lower through Tracr into a new model | 512/512 minority decisions correct |
| Patch original architecture using rank-one constraint compilation | 48 changed scalars |
| Initial bounded patch | 170/170 synthesis examples; fails on two length-9 oracle examples |
| Counterexample-guided refinement | 2 rounds, 172 synthesis examples; remaining length-9 oracle passed |
| Untouched final length-11 domain | 2,047/2,048 correct (99.951%) |
| Original model on the new minority task | 0/2,048 correct |
| Export/reload weight differences | 0 |
| Undo restores original hash | Yes |

There is still one final-domain failure; no refinement used that final result. The count-rule rewrite intentionally changes the task. All-position prediction disagreement and KL are large. This does not establish that existing language abilities were preserved or that a pretrained LLM gained reasoning capability.

The compiler uses observed hidden activations, LP constraint solving, and a rational check over recorded inequalities. It is not zero-data editing, not gradient descent, and not an IEEE floating-point equivalence proof. Every inference call and changed tensor is recorded.

[Full report, including failed first candidate](trained-rule-roundtrip.json).

## Ordinary pretrained language models

DistilGPT2 (81.9M parameters) and SmolLM2-135M (134.5M parameters) both passed independent program/native numerical checks after head scaling and selected-neuron direction removal, then checkpoint reload and undo tests. SmolLM2's eight-prompt smoke check had zero measured logit discrepancy on this environment. It changed 39,168 scalars across two matrices. No capability improvement is claimed for this edit.

Llama and Qwen2 adapters cover default RoPE, RMSNorm, grouped-query attention and gated MLPs. Synthetic adapter tests cover grouped KV heads, query biases, multiple edits and tied embedding reload. Unsupported variants fail explicitly.

Qwen2.5-0.5B-Instruct (494,032,768 parameters) also passed original/edited program-native checks on eight prompts, with zero measured logit error on this environment. Head scaling and selected-neuron direction removal changed 60,928 scalars. Export/reload introduced no weight differences, and undo restored the original hash. This tests the numerical backend on a third real architecture checkpoint, not an improvement in Qwen's reasoning.

[Qwen2 report](qwen2-smoke.json) · [SmolLM2 report](smollm2-smoke.json) · [DistilGPT2 report](distilgpt2-smoke.json).

## Editing agent experiments

Early local-policy studies used Qwen2.5-0.5B-Instruct as the editing agent. The first run produced invalid action shapes. After adding explicit syntax examples, both arms still failed to improve the minority objective in two steps. These failures are retained and are not evidence that a program interface outperforms a capable editing agent.

[First exploratory study](local-agent-v1.json) · [Protocol-revised study](local-agent-v2.json).

The primary editing agent has now been changed to a separate, authenticated **Codex TUI**, while the edited target remains local **SmolLM2-135M**. tmux and atomic request/response files connect Codex to the same inspection, trace, generation and evaluation controller. Separate standalone TUI sessions isolate direct/program experiments; session identity must be verified rather than relying on a shared-daemon active view.

The first eight-step hybrid Codex pilot kept development accuracy at 11/16 and reduced development NLL from 1.71258 to 1.68453. On 16 untouched completion probes, accuracy stayed 11/16 while NLL worsened from 1.65858 to 1.67353. Two head edits changed 73,728 scalars. Thus it is **not a demonstrated improvement**. The development examples are hand-authored exploratory probes, not a public benchmark or statistically meaningful superiority study.

The independent standalone direct-edit Codex run also kept development accuracy at 11/16. It changed two MLP neuron write vectors (1,152 scalars), reducing held-out NLL slightly from 1.65858 to 1.65662 without changing 11/16 accuracy. Collateral accuracy moved from 6/8 to 7/8, while all-position token disagreement was about 26% on target probes and 29% on collateral probes. These samples are far too small to establish a meaningful gain, and collateral distribution shifts must not be hidden by the task score.

The standalone program-only Codex run chose two attention-head scalings in layer 29, changing 73,728 scalars. Held-out accuracy again stayed 11/16; NLL was 1.63521. Its all-position disagreement was 1.45%, versus 26.1% for the direct candidate. Both used eight policy steps and 69 framework inference reservations under a maximum of 150. This exploratory single-seed result suggests a candidate tradeoff; it does not establish superiority. Prompt/context sizes and numerical tool interfaces differ by arm, and the hybrid run used an earlier workspace instruction template.

| Held-out 16-probe result | Original | Direct Codex | Program Codex | Hybrid Codex |
| --- | ---: | ---: | ---: | ---: |
| Accuracy | 11/16 | 11/16 | 11/16 | 11/16 |
| Target NLL | 1.65858 | 1.65662 | 1.63521 | 1.67353 |
| All-position disagreement | — | 26.1% | 1.45% | 0% |
| Changed scalars | 0 | 1,152 | 73,728 | 73,728 |

[Codex program summary](codex-program.json) · [Codex direct summary](codex-direct.json) · [Codex hybrid summary](codex-hybrid.json). The full controller journals and source checkpoints from that study remained in the local artifact directory. The tmux workspaces for those runs were on a machine-local temporary filesystem; that location is provenance for the recorded study, not an install path or a runtime default. Final/collateral examples are not sent to the policy. The file bridge is not a security sandbox. Cumulative Codex token usage can be collected from that workspace's own rollout with `weightscript codex-usage`; debugging turns must not be mistaken for experimental usage.

## Interpretation

The framework can now execute both prior-method backends, modify a learned task rule, compile into a new Transformer or patch the existing architecture, and run a genuine separate Codex as the editing agent. It still has restricted symbolic lowering, incomplete upstream paper reproduction, and no established superiority over the motivating research. Failure records and the remaining acceptance criteria are part of the deliverable.
