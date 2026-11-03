# Research target

The target is an agent that edits trained Transformers more reliably through a numerical/program interface than by manipulating raw tensors. The current implementation supplies extraction, inspection, a limited edit language, compilation, undo, and evaluation. It does not establish superiority.

The weight-space-programming post reports that agents can inspect and modify trained networks directly. Its strongest language-model result is a sparse MLP direction correction on Qwen3.5-0.8B; its Gemma result also shows that removing a failure mode is different from adding a missing capability. A fair comparison must distinguish these cases.

## Experiments to run next

1. Choose several trained GPT-2 checkpoints and pre-register target tasks, collateral-behavior datasets, candidate budgets and evaluation seeds. Use sufficiently challenging models/tasks to avoid the binary-majority pilot's ceiling accuracy.
2. Give the same external agent either raw tensor access or the extracted program/circuit tools. Match context, tool, inference and wall-time budgets. Neither arm sees held-out data or uses optimization/training.
3. Include untouched, random-edit, simple head/neuron scaling and global direction-removal controls. The existing matched-intervention pilot checks that changing interface alone does not change execution.
4. Compare target performance, collateral task performance, KL, edited parameter fraction, invalid-edit rate and time to a valid checkpoint. Repeat over agent/model/task seeds. Report unsuccessful runs and uncertainty.
5. Add a separate exact-adapter test for each new architecture before evaluating edits on it. Qwen3.5's architecture cannot be treated as a GPT-2 checkpoint. Porting the compiler also requires respecting projection orientation, grouped heads, rotary embeddings, normalization and possible hybrid layers.
6. Only then compare improvement per proposal/inference budget with a reproduction of the blog's baseline. Scores from a tiny majority predictor and MATH-500 are not comparable.

A plausible advantage is compositional editing with explicit circuit provenance and a compiler that checks whether an edit maps into the existing architecture. A smaller number of edited weights alone does not establish better behavior or preservation.

## Guarantees currently available

The tool validates checkpoint fingerprints and the supported projection structure, preserves shape/configuration/tying, records untouched tensors, and can restore the original checkpoint hash. The independent program uses actual LayerNorm, softmax and float32 computation rather than the LayerNorm linearization assumption in D-RASP extraction. Agreement tests are measured numerical evidence, not a formal equivalence theorem.

The integer Rust fixture is useful for exact exhaustive checks and compiler regression tests. It provides no evidence that general trained Transformers can be decompiled into short algorithms.

## Sources

- [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) — motivating agent experiments and baseline limitations.
- [Decompiling Transformers](https://lacoco-lab.github.io/home/decompiling_transformers/) — algorithm extraction, approximation/pruning tradeoffs, and trained checkpoints.
- [Tracr](https://github.com/google-deepmind/tracr) — forward RASP compilation; not a compiler into arbitrary existing checkpoint layouts.
