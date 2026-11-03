# Initial measured results

These runs establish working trained-checkpoint editing and compiler controls. They do not demonstrate that the program interface outperforms an agent directly editing tensors.

## Ordinary pretrained language model

Model: `distilbert/distilgpt2`, revision `2290a62682d06624634c1f46a6ad5be0f47f38aa`, 81,912,576 parameters. CPU float32 inference.

The edit scaled attention head 0 in layer 0 by 0.5 and removed half of one residual direction from four MLP write rows in layer 3. The direction was the original write vector of neuron 0. This is an integration test, not an improvement proposal.

| Check | Result |
| --- | --- |
| Original extracted program vs native inference, 8 prompts | Passed |
| Edited program vs compiled native inference, 8 prompts | Passed |
| Maximum original logit absolute error | 0.0000458 |
| Maximum edited logit absolute error | 0.0000610 |
| Comparison tolerance | `abs(a-b) <= 2e-5 + 2e-5 * abs(b)` |
| Changed scalars | 52,224 across two matrices |
| Exported/reloaded checkpoint weight differences | 0 |
| Undo restored original checkpoint SHA-256 | Yes |
| Gradient updates during editing | 0 |

The maximum absolute errors exceed the absolute tolerance alone; the elementwise absolute-plus-relative comparison passes. This is not bitwise or formal equivalence. Eight prompts are an integration check, not a behavioral evaluation set.

[Machine-readable report](distilgpt2-smoke.json). Reproduce with `python examples/language_smoke.py`.

## Gradient-trained binary-majority checkpoint

Source: [upstream checkpoint](https://github.com/lacoco-lab/decompiling_transformers/tree/787398b0f4393a90ad02cd8e71be48d1e85d2b21/share/saved_models/bin_majority-%401l1h16d3lr01drop). One layer, one head, width 16, 5,872 parameters. No model training was performed here.

Development: 96 generated examples, bit-string lengths 15, 31, 63. Held-out: 96 separately generated examples, lengths 17, 33, 65. Generation seeds are 710 and 928. BOS=2 and SEP=3 follow upstream's tokenizer. Labels are the majority bit; all tested lengths are odd.

Both arms proposed the same eight interventions: untouched baseline, then head output scales 0, 0.25, 0.5, 0.75, 1.25, 1.5 and 2. The direct arm specifies matrix rows; the program arm specifies a head. Each used 24 development forward calls under a common maximum budget of 48. Selection used development accuracy, then target NLL. Both selected head scale 2. Held-out data was evaluated only after selection within each run.

| Held-out metric | Original | Direct tensor control | Program control |
| --- | ---: | ---: | ---: |
| Next-token accuracy | 100% | 100% | 100% |
| Target NLL | 0.018926 | 0.001628 | 0.001628 |
| Changed weight scalars | 0 | 256 | 256 |
| All-position token disagreement vs original | — | 3.228% | 3.228% |
| Reference KL per position | — | 0.18172 | 0.18172 |

Original and edited program/native logit differences were zero on these 96 held-out examples. Export/reload introduced zero weight differences. Confidence improved on this sample, but accuracy was already saturated. Internal-position predictions changed, so this is not a blanket behavior-preservation claim. There are no statistical confidence intervals or multiple agent seeds in this pilot.

The equivalent controls tie, as they should. These results provide no evidence of beating the weight-space-programming post or of superior proposal quality. A real comparison still needs independent agent arms, challenging tasks, collateral-behavior tests and matched proposal-generation costs.

[Benchmark report](majority-benchmark.json) · [Equivalence report](majority-equivalence.json). Reproduce with the fetch script and `examples/research_benchmark.py` described in the main README. A second run from the fetch script's hash-checked download reproduced the same metrics.
