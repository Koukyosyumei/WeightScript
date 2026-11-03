"""Published MATH-500 / GSM8K settings for the weight-space programming reproduction.

Sources, used only where they state a setting:

- Takehi, Teoh, Zhang, Khattab. "Sparks of Weight-Space Programming" (2026-11-02),
  https://rikiyat.github.io/blog/weight-space-programming/
- RikiyaT/lm_understand-logs exp8_math (scripts MIT; logs CC BY 4.0):
  code/evaluate.py protocols ``think35`` and ``nothink35``, code/profiles.py,
  code/mathqa.py, and exp8_math/README.md.
- Qwen3.5-0.8B model card, as cited by those sources: thinking-mode sampling
  temperature 0.6, top_p 0.95, top_k 20, presence_penalty 1.5; non-thinking
  temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5; math instruction
  "Please reason step by step, and put your final answer within \\boxed{}."

Settings the public sources do not state are left unset. They are not filled in.
"""

from __future__ import annotations

from dataclasses import dataclass


FORMAT_VERSION = 1

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

# evaluate.py prepares these Hugging Face splits. It does not pin a dataset revision.
MATH500_HF = {"repo": "HuggingFaceH4/MATH-500", "config": None, "split": "test"}
GSM8K_HF = {"repo": "openai/gsm8k", "config": "main"}

# Raw split sizes the loaders above return. The blog's 7,469 / 7,275 practice
# counts are after a decontamination pass whose code is not in the public release.
PUBLISHED_COUNTS = {
    ("math500", "test"): 500,
    ("gsm8k", "test"): 1319,
    ("gsm8k", "train"): 7473,
}

# evaluate.py SAMPLES. The blog table uses these for the think35 runs.
PUBLISHED_SAMPLES = {"math500": 4, "gsm8k": 2}

# README: public Qwen/Qwen3.5-0.8B weights this experiment diffs against.
QWEN35_BASE_MODEL = "Qwen/Qwen3.5-0.8B"
QWEN35_BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"

# mathqa.py pins the checker it calls. A different installed version is reported,
# not treated as 0.9.0.
MATH_VERIFY_VERSION = "0.9.0"
MATH_VERIFY_REQUIRES = (
    "math-verify==0.9.0",
    "latex2sympy2_extended==1.11.0",
    "antlr4-python3-runtime==4.13.2",
)

DECLARED_LIMITS = (
    "Primary correctness is math-verify on the last closed \\boxed{} (or \\fbox{}) after the last </think>. hendrycks is_equiv is stored separately and is not the article accuracy.",
    "If math-verify is not installed, correct is null when a box was extracted. is_equiv does not fill that field.",
    "An unclosed final box is an extraction failure even if an earlier box closed. The published extractor does not fall back to the previous box.",
    "When both \\\\boxed and \\\\fbox occur, the last \\\\boxed wins even if an \\\\fbox follows it. That is last_boxed in mathqa.py.",
    "The blog's hit-token-limit column is the share of samples with finish_reason 'length'. No other stop reason is counted as a cut.",
    "verbatim_loop_end is mathqa.loop_signals. repetitive_end follows the rule in that docstring; mcqa.loop_signals itself is not in the public exp8 release.",
    "The 13-word practice decontamination (practice.grams / practice.norm) is not in the public code. This harness does not approximate it. It keeps evaluation and practice splits apart and can reject exact normalized-text overlap.",
    "Accuracy is the mean over samples, not a vote and not pass@k.",
    "Wall time is checked between examples and accumulates across resumes of one results file. One generate call is not preempted.",
    "A resumed example is regenerated as a whole. Completed samples are not mixed with a later draw.",
    "The only evaluation subset is a prefix of canonical dataset order, fixed before generation by max_examples. There is no score-based or stride selection.",
    "vLLM, bfloat16, and the engine seed are caller-declared. This package does not load a model.",
    "Gemma's lenient headline is acc_lenient. The Qwen think35 headline is strict boxed accuracy.",
    "No LLM judge is called.",
)


@dataclass(frozen=True)
class Protocol:
    """One published generation and grading profile."""

    name: str
    thinking: bool
    sampling: dict
    samples: dict
    instruction: str
    headline: str
    lenient_headline: bool
    engine: str | None
    dtype: str | None
    source: str

    @property
    def max_new_tokens(self) -> int:
        return int(self.sampling["max_tokens"])

    def samples_for(self, dataset: str) -> int:
        try:
            return int(self.samples[dataset])
        except KeyError as error:
            raise KeyError(
                f"protocol {self.name!r} publishes no sample count for dataset {dataset!r}"
            ) from error


def _sampling(payload: dict) -> dict:
    return dict(payload)


# evaluate.py SAMPLING_THINK35. Blog Experiment 8 table.
THINK35 = Protocol(
    name="think35",
    thinking=True,
    sampling=_sampling(
        {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 1.5,
            "max_tokens": 32768,
            "seed": 0,
        }
    ),
    samples=dict(PUBLISHED_SAMPLES),
    instruction=INSTRUCTION,
    headline="acc",
    lenient_headline=False,
    engine="vllm",
    dtype="bfloat16",
    source="exp8_math/code/evaluate.py SAMPLING_THINK35 and README evaluation paragraph",
)

# evaluate.py SAMPLING_NOTHINK35. The blog table is not this profile.
NOTHINK35 = Protocol(
    name="nothink35",
    thinking=False,
    sampling=_sampling(
        {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "presence_penalty": 1.5,
            "max_tokens": 8192,
            "seed": 0,
        }
    ),
    samples=dict(PUBLISHED_SAMPLES),
    instruction=INSTRUCTION,
    headline="acc",
    lenient_headline=False,
    engine="vllm",
    dtype="bfloat16",
    source="exp8_math/code/evaluate.py SAMPLING_NOTHINK35; Qwen3.5-0.8B non-thinking model-card setting as cited there",
)

# evaluate.py SAMPLING_GEMMA3 and profiles.py gemma270 lenient headline.
# The blog's Gemma table is MATH-500 only. GSM8K still has the evaluator's sample count.
GEMMA3 = Protocol(
    name="gemma3",
    thinking=False,
    sampling=_sampling(
        {
            "temperature": 1.0,
            "top_k": 64,
            "top_p": 0.95,
            "max_tokens": 8192,
            "seed": 0,
        }
    ),
    samples=dict(PUBLISHED_SAMPLES),
    instruction=INSTRUCTION,
    headline="acc_lenient",
    lenient_headline=True,
    engine=None,
    dtype=None,
    source="exp8_math/code/evaluate.py SAMPLING_GEMMA3; profiles.py gemma270 lenient grading. Engine and dtype are not stated for this profile.",
)

PROTOCOLS = {item.name: item for item in (THINK35, NOTHINK35, GEMMA3)}


def protocol(name: str = "think35") -> Protocol:
    try:
        item = PROTOCOLS[name]
    except KeyError as error:
        known = ", ".join(sorted(PROTOCOLS))
        raise KeyError(f"unknown protocol {name!r}; known: {known}") from error
    return item


def user_turn(problem: str, *, instruction: str = INSTRUCTION) -> str:
    """User-message body: the problem, a newline, and the published instruction."""

    if not isinstance(problem, str) or problem.strip() == "":
        raise ValueError("problem must be a non-empty string")
    return problem + "\n" + instruction


def render_chat(tokenizer, problem: str, *, thinking: bool, instruction: str = INSTRUCTION) -> str:
    """Apply the checkpoint's own chat template.

    The published runner calls ``apply_chat_template(..., enable_thinking=thinking)``.
    A tokenizer that cannot take ``enable_thinking`` is rejected. A template string
    is not invented here.
    """

    messages = [{"role": "user", "content": user_turn(problem, instruction=instruction)}]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
    except TypeError as error:
        raise ProtocolError(
            "tokenizer.apply_chat_template does not accept enable_thinking. "
            "think35 requires the Qwen3.5 chat template. No fallback template is applied."
        ) from error
    if not isinstance(rendered, str) or rendered == "":
        raise ProtocolError("apply_chat_template did not return a non-empty string")
    return rendered


class ProtocolError(ValueError):
    """The call would not follow a published protocol."""
