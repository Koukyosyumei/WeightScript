# MATH-500 and GSM8K reproduction harness

This harness scores an untouched checkpoint and an edited checkpoint on the
math evaluation from Experiment 8 of [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).
It does not edit weights, load Qwen, or start vLLM. Those stay with the main
integration. The public sources for every setting below are
`exp8_math/code/evaluate.py`, `exp8_math/code/mathqa.py`, `exp8_math/code/profiles.py`,
and `exp8_math/README.md` in [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs)
(scripts MIT; logs and data CC BY 4.0). Third-party benchmark text stays under
its own license.

## What is fixed

The blog table uses protocol `think35`.

| Setting | Value |
| --- | --- |
| Base model | `Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17` |
| Thinking | on, through the checkpoint's own chat template (`enable_thinking=True`) |
| User text | problem, newline, `Please reason step by step, and put your final answer within \boxed{}.` |
| Sampling | temperature 0.6, top_p 0.95, top_k 20, presence_penalty 1.5, seed 0 |
| New tokens | 32,768 |
| Engine declared by the article | vLLM, bfloat16 |
| MATH-500 | `HuggingFaceH4/MATH-500` test, 500 rows, 4 samples |
| GSM8K | `openai/gsm8k` config `main` test, 1,319 rows, 2 samples |
| GSM8K gold | text after `####`, commas removed |
| Score | mean over samples of the last closed box, checked with math-verify 0.9.0 |
| Hit token limit | share of samples whose finish reason is `length` |

`nothink35` is the model-card non-thinking setting the same evaluator records
(temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5, 8,192 tokens,
thinking off). It is not the blog table. `gemma3` is the additional Gemma
experiment (temperature 1.0, top_k 64, top_p 0.95, 8,192 tokens, no presence
penalty in the published sampling dict). Its headline is the lenient reader.
The Qwen headline is not lenient.

The dataset loaders in the article do not pin a dataset revision. This harness
does not invent one.

## Grader

Primary equivalence is math-verify, called as the published grader calls it:

```text
parse("\boxed{" + answer + "}")
verify(gold_parse, pred_parse)
```

The pinned stack from `mathqa.py` is `math-verify==0.9.0`,
`latex2sympy2_extended==1.11.0`, and `antlr4-python3-runtime==4.13.2`.
Install that stack in the evaluation environment when a blog-table number is
required. This package does not install it and does not import it until a
boxed answer is graded.

If math-verify is missing, or its `parse` / `verify` signature does not match,
`correct` is null. Hendrycks `is_equiv` still fills `equiv`. That field is the
original MATH string checker. It is not the article accuracy, and it is never
copied into `correct`.

A generation with no closed final box is incorrect without calling math-verify.
An unclosed final `\boxed` does not fall back to an earlier box. If both
`\boxed` and `\fbox` appear, the last `\boxed` wins even when an `\fbox`
follows it. Extraction reads the text after the last `</think>`.

The Hendrycks helpers and the boxed reader are taken from the MIT-licensed
`exp8_math/code/mathqa.py`, which attributes `is_equiv` to hendrycks/math (MIT).
The copyright notice is in `python/weightscript/math_benchmark/extract.py`.

## Loops

The blog column "Hit token limit" is `cut_share`: finish reason `length`.
No other finish reason counts as a cut. A missing finish reason stays null,
and a reason other than `stop` or `length` is not stored.

`verbatim_loop_end` is `mathqa.loop_signals`: drop the last word, then look for
one block of 1 to 200 words repeated at least three times and spanning at least
40 words. `repetitive_end` follows the rule in that docstring (in the last 200
words, at least half of the 4-word positions are a phrase that occurs at least
three times). `mcqa.loop_signals`, which the docstring calls, is not in the
public release, so `repetitive_end` is that stated rule and not a line-by-line
port of the missing helper.

## Splits

Evaluation accepts MATH-500 test and GSM8K test. Practice accepts GSM8K train.
A train split cannot be scored as the benchmark. `run_evaluation` refuses a
practice set.

The article drops a practice problem when a 13-word window, or the problem's
normalized text, occurs in an evaluation problem (`practice.grams` /
`practice.norm`). Those functions are not in the public code. This harness
does not approximate them. `assert_disjoint` rejects exact NFKC/casefold
overlap and says that it is not the 13-word filter. The blog's decontaminated
practice counts are 7,469 GSM8K and 7,275 MATH. The raw GSM8K train split is
7,473. Do not treat a raw train file as the decontaminated practice set.

The only scored subset is a prefix of canonical order, chosen with
`max_examples` before generation. There is no stride, no id filter, and no
filter on which items an arm got wrong.

## Where files go

`load_hf(..., cache_dir=...)` writes the datasets cache there. With no
argument, an existing `HF_DATASETS_CACHE` is left to the datasets library.
If that is also unset and `WEIGHTSCRIPT_CACHE_DIR` is set, the cache is
`<WEIGHTSCRIPT_CACHE_DIR>/datasets`. Otherwise Hugging Face's own default
is used. `WEIGHTSCRIPT_RUNS_DIR` is the directory for result JSONL when a
caller wants one; `run_evaluation` still writes only to the path it is given.
`fetch` downloads one published split into `--dest` or into the resolved
cache directory, and refuses to start when neither is set. Downloads are
capped at 32 MiB. GSM8K's published files are parquet and are converted only
when `pyarrow` is installed. The article loaders do not pin a dataset
revision, and this package does not invent one. A fetched JSONL is a local
file, so `blog_table_settings` stays false for it. One GSM8K training
problem contains U+2028. The writer escapes that character, and the loader
returns it unchanged.

No default path names a particular machine or disk.

## Results file

`run_evaluation` writes JSONL. The first record is the config and its hash.
Later records are one sample each (raw text, gold, extraction, both checkers,
finish reason, token count, loop flags) or an invocation record. The file is
bound to the source model, source revision, checkpoint id, fingerprint, arm,
dataset bytes, protocol, effective sampling, grader version, and budgets.
Reopening it with a different binding raises `ConfigMismatch` and does not
change the file.

A finished example is skipped on resume. A crash mid-example drops that
trailing example and regenerates all of its samples together. Completed
samples are not mixed with a later draw. A hole that is not the trailing
example is an error, and the file is left as it is.

`walltime_s` is checked between examples and summed across resumes of the same
file. One `generate` call is not preempted. A lower token or sample cap is
stored as a deviation. A cap above the published value is rejected.

## Comparison

`compare_runs(untouched_path, edited_path)` allows the arm, checkpoint id, and
fingerprint to differ. Budgets, dataset identity, protocol, grader, rendering
mode, and the scored id list must match. `delta_acc` is set only when both
runs finished every configured example and both accuracies are numbers.
Incomplete runs and non-prefix id lists do not get a delta. Subtracting the
arm summaries of an incomplete or incomparable pair is not a protocol result.

`blog_table_settings` is the only flag that means the blog table's settings
were met. It is false for a local JSONL stand-in, a prefix, a sample or token
deviation, a missing or different math-verify version, a chat template this
process did not apply, or a caller declaration other than vLLM, bfloat16,
`Qwen/Qwen3.5-0.8B`, and the published revision. Engine, dtype, and the vLLM
constructor seed are caller-declared. This package cannot see them.

`acc_equiv` and `acc_lenient` are always separate from `acc`.

## Main integration API

```python
from weightscript.math_benchmark import (
    Budgets,
    RunSpec,
    compare_runs,
    load_hf,
    load_jsonl,
    render_chat,
    run_evaluation,
    write_summary,
)

examples = load_jsonl(path, dataset="math500", split="test", role="evaluation")
# or: examples = load_hf("math500", split="test")  # imports datasets; downloads

spec = RunSpec(
    source_model="Qwen/Qwen3.5-0.8B",
    source_revision="2fc06364715b967f1860aea9cf38778875588b17",
    checkpoint_id="untouched",          # a different id on the edited arm
    checkpoint_fingerprint="<weight hash>",
    arm="untouched",                    # or "edited"
    protocol_name="think35",
    engine="vllm",                      # caller declaration
    dtype="bfloat16",
)

def generate(requests):
    # requests for one example, sample_index 0..n-1, same prompt.
    # Keys: example_id, sample_index, index, user_content, rendered_prompt,
    # sampling, max_new_tokens, seed, protocol.
    # There is no gold, answer, or solution key.
    prompt = requests[0]["rendered_prompt"]
    sampling = requests[0]["sampling"]
    # Article engine: LLM(..., dtype="bfloat16", seed=0) and
    # SamplingParams(n=len(requests), **sampling). Map presence_penalty
    # yourself. Do not drop it.
    raise NotImplementedError

summary = run_evaluation(
    examples,
    spec,
    generate,
    "results/untouched.jsonl",
    tokenizer=tokenizer,   # must accept enable_thinking; otherwise pass render=None
                           # and put the template in the generator
)
report = compare_runs("results/untouched.jsonl", "results/edited.jsonl")
# Quote report["delta_acc"] only when report["complete"] and report["comparable"].
# Quote it as the blog table only when report["blog_table_settings"] is true.
```

Use the same `Budgets` and the same loaded dataset for both arms. Set
`max_examples`, `max_samples`, `max_new_tokens`, and `walltime_s` before the
first example is generated. Changing them later is a different results file.

`python examples/math_benchmark.py protocol` prints a protocol.
`run` grades a JSONL of generations and does not call a model.
`compare` exits 2 when the pair is not a finished match.
`smoke` writes a two-problem canned pair under `--out` and is not a score.
`fetch --dataset-name math500 --split test --dest DIR` saves a local snapshot.
It does not download a model.

Declared limits that are not bugs are listed in
`weightscript.math_benchmark.DECLARED_LIMITS` and in the sections above.
No LLM judge is used.
