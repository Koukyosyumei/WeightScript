# Decompilation API and measured limits

This module runs the pinned [Decompiling Transformers](https://lacoco-lab.github.io/home/decompiling_transformers/) component, full-path, and QK engines locally, matches direct embedding QK contractions against that repository's attention primitives, and lowers explicit semantic rules into an existing GPT-2. It is a bounded integration. A separate driver runs the pinned AdamW mask sampler and tests MLP primitives under a hard step cap. It does not run the paper's HTCondor sweep, the 500–1000 step schedule, or `greedy_then_round` attention replacement.

Upstream source stays unmodified. The driver records these compatibility adapters:

- Import supplies the missing `my_paths` directory constant.
- Optimal-ablation vectors keep attention output biases on the component engine.
- Cached positional embeddings are broadcast across QK batches.
- Empty auxiliary tensors replay when a stage has no query-bias rows.
- If the pinned full-path renderer references undefined heatmap labels, the same config is rendered with the upstream QK renderer. The artifact records when that fallback ran.

Callers pass the upstream checkout and the checkpoint. Public functions do not embed a machine path.

## Python hooks

```python
from weightscript.backends.upstream import decompile, execute_artifact
from weightscript.backends.upstream_stages import decompile_stages, execute_stages
from weightscript.backends.upstream_search import decompile_coefficient, execute_coefficient
from weightscript.backends.upstream_primitives import match_qk_primitives
from weightscript.backends.pipeline import disjoint_domains, summarize_stages, decompile_and_edit
from weightscript.backends.rules import validate_rule, rule_predict, supervised_pairs
from weightscript.symbolic import synthesize_rules, synthesize_sequence_rules, synthesize_binary, lower
from weightscript.synthesis import compile_rule, cegis, replay_patch
```

`decompile_stages(model, checkout, calibration, oracle)` runs all three engines with greedy zero-ablation. Mask search and LayerNorm variance fitting use the calibration rows. Oracle rows are scored after the masks are chosen. `decompile_coefficient` runs the same engines through the pinned AdamW mask sampler and auxiliary update, with a hard step cap, then tests upstream MLP primitives. `execute_stages`, `execute_artifact`, and `execute_coefficient` replay a serialized artifact without refitting. `summarize_stages` reads acceptance, edge counts, and primitive fits from either artifact.

`synthesize_sequence_rules` compares explicit `span_next` programs with teacher-forced model decisions. Calibration selects the rule. Evaluation and final rows are reports. Below the fidelity threshold the artifact output stays the opaque native model.

`compile_rule` and `cegis` patch the last MLP and the final LayerNorm bias of a private copy. The source checkpoint is fingerprinted before and after. Held-out rows are scored only after refinement stops. `cegis` also rescores the original oracle on the final patch and records how many oracle rows were absorbed into the linear program. A later round that exceeds the variable budget keeps the last solved patch and sets `refinement_stopped`.

The example `examples/trained_rule_roundtrip.py --suite --checkout CHECKOUT --output DIR` runs the three saved checkpoints `bin_majority-@1l1h16d3lr01drop`, `unique_copy-@2l1h64d3lr01drop`, and `sort-@1l1h256d3lr01drop` from `CHECKOUT/share/saved_models`. The same script with `--coefficient` runs the capped AdamW driver and writes a separate directory. Those example flags are separate from `weightscript decompile --engine three-stage`.

## CLI

Datasets are JSON lines. Each line is an object with nonempty integer `input_ids`.

```bash
weightscript decompile MODEL CALIBRATION OUTPUT --checkout CHECKOUT --engine three-stage
weightscript run-decompiled MODEL ARTIFACT '[2,0,1,3]' --checkout CHECKOUT --stage qk
weightscript synthesize MODEL CALIBRATION OUTPUT --candidates rules.json --sequence
weightscript rule-check RULE.json
weightscript rule-predict RULE.json '[8,1,1,0,9]'
weightscript rewrite MODEL RULE SYNTHESIS ORACLE OUTPUT --heldout HELD
weightscript pipeline MODEL OUTPUT --checkout CHECKOUT --candidates rules.json \
  --rule RULE.json --synthesis SYNTHESIS --oracle ORACLE --final FINAL
```

`--engine component` remains the default and writes `program.txt` beside `artifact.json`. `--engine three-stage` writes `artifact.json` and `summary.json`. `--stage component` can replay that summary's nested component artifact. `pipeline` writes `report.json` and exports a checkpoint only when the causal stage and the behavioral hypothesis are both accepted. A rejected run leaves the report in place and returns status 1.

## Normalization assumptions

- Positions are `0 .. length-1`. Upstream training also sampled random position offsets. These measurements do not resample those offsets.
- Stage engines replace native LayerNorm with a per-vertex log-variance fit on the calibration rows.
- Greedy artifacts saved before the coefficient pilot score stage acceptance on the final logit of each row. In that suite the rows end at the separator, so the final logit is the first answer token. Those files are unchanged.
- Current drivers score every requested answer position. A row with `answer_start` requires argmax agreement at each logit that predicts a token from that index through the end of the sequence. A failed initial linear-LayerNorm agreement rejects the stage. The token after the last answer is not a requested position.
- `logit_reconstruction` is true only when the maximum absolute logit error is at most `1e-4`. Every trained stage below fails that bar. Decision agreement and logit equality are different claims.
- QK primitive matching row-centers the weight contraction because softmax ignores a constant per query. A relative residual above `0.05`, a row-constant contraction, or a path that is not a pair of embedding matrices stays opaque. A numerical fit is not a replacement-fidelity proof.

Semantic rules are validated data, not evaluated code. `span_next` kinds are `copy`, `sorted`, and `majority`. Selection `which` is `first`, `last`, `min`, or `max`. A branch condition token of `0` takes the else branch.

## Trained measurements

Pinned upstream revision `787398b0f4393a90ad02cd8e71be48d1e85d2b21`. Two CPU threads. Source weight hashes were unchanged after every task. Greedy edge search visited every edge. Replay of the saved majority artifact reproduces the component decision on `2 0 0 0 1 0 1 1 3` and the full-path/QK disagreement on that same row.

Domains are disjoint. Majority uses every non-tie binary string of length 1, 3, and 5 (42) for calibration, length 7 (128) for oracle, and length 9 (512) for final. Copy and sort share one content sample: 8 strings at each of lengths 2, 3, and 4 (24), 16 strings of length 6, and 16 strings of length 8, drawn without replacement from token ids `0..149` with seeds 11, 22, and 33. Markers are the checkpoint special tokens.

| Task | Native final positions | Behavioral rule | Component | Full path and QK |
| --- | --- | --- | --- | --- |
| Majority | 512/512 | `majority`, 42/42, 128/128, 512/512 | Accepted. 13 edges to 3. Oracle and final replay 128/128 and 512/512 | Rejected. Oracle replay 93/128. Final replay 386/512. QK has 0 edges. Rendered program is uniform token aggregation into an opaque MLP |
| Unique copy | 144/144 | `copy`, 96/96, 112/112, 144/144 | First-token acceptance. 32 edges to 8. Full-span replay 110/112 oracle and 129/144 final | First-token acceptance only. Full-span replay 18/112 and 21/144. Edges 8 to 6, then 3 to 2 |
| Sort | 143/144 positions, 15/16 sequences | Rejected. Best calibration agreement 92/96 | First-token acceptance after pruning, while the initial linear LayerNorm engine agrees on 22/24. Full-span replay 104/112 and 131/144 | Rejected. Full-path calibration 2/24 and oracle 0/16. QK calibration 19/24 and oracle 11/16. Full-span QK replay 50/112 and 58/144 |

Majority foils on the calibration set were copy 30/42 and sorted 24/42. Copy foils were sorted 20/96 and majority 0/96. The sort rule matches the task labels on every scored position and matches the model on 92/96, 110/112, and 143/144. Calibration and oracle are below 0.99, so the synthesizer keeps the native model and does not install the rule. Final agreement is not allowed to overturn that choice.

Direct embedding QK matching accepted no library primitive. Majority's only attention node is uniform selection because the masks removed every QK and affine-key term. Copy's remaining contraction is `wte` with `attn_output-0-0-wte`, which has no static embedding matrix and stays opaque. Sort's `wte`/`wte` contraction's closest primitive is `every_second` (`k%2==q%2==0`) at relative residual 0.976, above the 0.05 tolerance, so it stays opaque. This greedy suite does not replace MLP or unembedding primitives. The coefficient pilot below tests the upstream MLP converter on its own masks.

Maximum absolute logit error on the accepted component stages was about 5.0 (majority), 14.6 (copy), and 18.8 (sort).

## Semantic edits

Edits use a private copy. Held-out rows never enter the linear program.

- Majority inverted across 0.5, rank-one patch, 48 changed scalars in `transformer.h.0.mlp.c_proj.weight` and `transformer.ln_f.bias`. Two oracle rows were absorbed. Oracle rescore 128/128. Held-out length 9 is 510/512. The two misses are the all-zero and all-one strings, where the patch still predicts the majority bit.
- Copy checkpoint patched toward `sorted` on four prefixes. Oracle rescore is 4/4 after all four oracle prefixes were absorbed. Four held-out prefixes score 0/4.
- Sort checkpoint patched toward `copy` on four prefixes (synthesis 4/4, oracle 0/4). The next round asks for 8200 variables, 8 snapshots, rank 8, and 10,036,800 projected entries, above the 5,000,000 entry budget. Refinement stops. Oracle rescore and held-out are 0/4. The patch does not move the first answer token from the sorted value to the copied value.

`compile_rule` also lowers `select_token` with `which=min` and a `span_next` copy on a tiny random GPT-2 in the tests. Those checks are compiler tests, not trained-checkpoint results.

Serialized reports and replayable stage artifacts are in `docs/results/decompilation/`. Stage `accepted` in those greedy files uses the driver's first-answer-token metric. The full-span replay numbers in each summary are the sequence measurement. Those rejection files were not overwritten by the coefficient pilot.

## Bounded coefficient search

`examples/trained_rule_roundtrip.py --coefficient` runs `decompile_coefficient` on the same three checkpoints and the same content seeds. Rows are full sequences. `answer_start` is the first token after the separator, so majority has one answer position per row and copy/sort include every copied or sorted token plus eos. The optimizer is the pinned `MaskSampler` AdamW loop: soft masks, sparsity penalty `1e-3`, KL on the requested answer positions, gradient clip 5, binary commit at logit 0, and auxiliary AdamW on the ablation vectors and log-variances. Settings for this pilot were 16 joint steps, 4 pretrain steps on the later stages, batch size 8, and a 180 second cap. The paper's countdown extension, baseline-loss stop, 500-step pretrain, and HTCondor sweep were not run. Full-path output vectors were warm-started from the previous stage, and the QK affine query was warm-started from the full-path query vector before AdamW. Both adaptations are recorded on the artifact.

Acceptance requires the initial linear engine and the committed mask to match every requested answer position on calibration and on the oracle. Final rows are replayed afterwards and do not select masks or primitives. Logit reconstruction remains false. Source hashes were unchanged. The MLP candidates in this driver are `erase`, `keep_one` of each traced input, and `combine`. That list is not the longer primitive set in upstream `convert_mlp.py`.

| Task | Component | Full path | QK | MLP primitive |
| --- | --- | --- | --- | --- |
| Majority | Accepted. 13 edges to 11. Oracle 128/128. Final replay 512/512. Max logit error about 17.8 | Accepted. 13 to 12. Oracle 128/128. Final 512/512. Max logit error about 21.4 | Accepted. 2 edges to 1. Oracle 128/128. Final 512/512. Max logit error about 8.9 | Installed `keep_one` of `attn_output-0-0-wte`. FVU 0.288 on 42 rows. Causal, oracle, and length-9 replay 128/128 and 512/512 |
| Unique copy | Accepted on the answer span. 32 edges to 29. Calibration 96/96, oracle 112/112, final 144/144. Max logit error about 12.2 | Rejected. Initial answer agreement 95/96, so the linear engine already misses the bar. Oracle 108/112. Final replay is 144/144 and does not overturn that rejection. 35 edges to 34 | Rejected. Initial agreement 95/96. Oracle 102/112. Final 140/144, sequences 12/16. 26 edges stay 26 | Not installed. Best `keep_one` of `wte` has FVU 0.011 but oracle agreement 54/112. Layer 1 stays opaque because layer 0 was not converted |
| Sort | Rejected. Initial answer agreement 95/96, below 0.99, and no edge was removed. Oracle 106/112. Final 139/144. Max logit error about 22.7 | Rejected. Initial 14/96. Oracle 15/112. Final 12/144. 15 edges to 14 | Rejected. Initial 28/96. Oracle 17/112. Final 16/144. 4 edges stay 4 | Not installed. Best `keep_one` has FVU 0.018 and matches the unreplaced stage on 93/96 calibration positions, but matches the frozen model on 27/96 and 8/112 |

Majority's single answer bit is the whole task, so 512/512 is every length-9 decision, not a multi-token span. Copy's accepted component stage is the multi-token result. The later copy and sort stages stay rejected. A low FVU is not replacement fidelity: copy's 0.011 fit and sort's 0.018 fit were both refused. The greedy artifacts in the parent directory still record majority full-path/QK at 386/512 and copy later-stage full-span replay at 21/144.

Reports are in `docs/results/decompilation/coefficient-pilot/`.

## What remains open

The capped coefficient search does not reproduce the paper's sparse circuits or its training schedule. Mask logits were still inside (-1, 1) when the step cap stopped the run, and most edges remained. Logit-level reconstruction is open on every trained stage. Copy after the component stage, and sort at every stage, miss the 0.99 answer-span bar. Unknown QK paths and rejected MLP fits stay opaque. A finite behavioral library is labeled as behavioral synthesis.
