"""Check the published Weight-Space Programming math edits, then optionally smoke-sample.

This does not reproduce the blog's MATH-500 or GSM8K scores. The published
protocol is vLLM, thinking on, temperature 0.6, top_p 0.95, top_k 20,
presence_penalty 1.5, 32768 new tokens, four MATH-500 samples and two GSM8K
samples, graded by math-verify on the last boxed answer. A run here rebuilds
the published diffs against Qwen/Qwen3.5-0.8B at revision
2fc06364715b967f1860aea9cf38778875588b17 and, with --generate, draws one short
CPU sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
PROTOCOL_SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 1.5,
    "max_new_tokens": 32768,
    "seed": 0,
    "thinking": True,
}


def _header(path: Path):
    with path.open("rb") as handle:
        nbytes = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(nbytes))
    return header, 8 + nbytes


def _load_named(path: Path, names: set[str]):
    header, offset = _header(path)
    out = {}
    with path.open("rb") as handle:
        for name, info in header.items():
            if name == "__metadata__" or name not in names:
                continue
            start, end = info["data_offsets"]
            handle.seek(offset + start)
            raw = np.frombuffer(handle.read(end - start), dtype=np.uint16).copy()
            out[name] = raw.reshape(info["shape"])
            if info["dtype"] != "BF16":
                raise ValueError(f"{name} is {info['dtype']}, expected BF16")
    missing = names - set(out)
    if missing:
        raise ValueError(f"missing tensors: {sorted(missing)[:4]}")
    return out


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    raw = (
        np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    )
    raw = raw + 0x7FFF + ((raw >> 16) & 1)
    return (raw >> 16).astype(np.uint16)


def _rebuild(base_bits: np.ndarray, form: str, bundle, index: int) -> np.ndarray:
    if form == "lowrank":
        weights = _bf16_to_f32(base_bits).astype(np.float64)
        left, right = bundle[f"t{index}_a"], bundle[f"t{index}_b"]
        for rank in range(left.shape[1]):
            weights = weights + np.outer(left[:, rank], right[rank])
        edited = _f32_to_bf16_bits(weights.astype(np.float32)).reshape(-1)
    else:
        edited = base_bits.reshape(-1).copy()
    edited[bundle[f"t{index}_idx"]] = bundle[f"t{index}_val"]
    return edited.reshape(base_bits.shape)


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def verify_edits(weights: Path, edits_root: Path) -> dict:
    report = {}
    started = time.perf_counter()
    for run in ("p10c", "p25c", "p50c"):
        edit_dir = edits_root / run / "edit"
        manifest = json.loads((edit_dir / "manifest.json").read_text())
        bundle = np.load(edit_dir / "edit.npz")
        names = {item["name"] for item in manifest["tensors"]}
        tensors = _load_named(weights, names)
        checked = []
        for index, item in enumerate(manifest["tensors"]):
            base = tensors[item["name"]]
            if list(base.shape) != item["shape"]:
                raise ValueError(
                    f"{item['name']} shape {base.shape} != {item['shape']}"
                )
            base_hash = _sha256(base)
            if base_hash != item["base_sha256"]:
                raise ValueError(f"{run} {item['name']} base hash mismatch")
            edited = _rebuild(base, item["form"], bundle, index)
            edited_hash = _sha256(edited)
            if edited_hash != item["edited_sha256"]:
                raise ValueError(f"{run} {item['name']} edited hash mismatch")
            checked.append(
                {
                    "name": item["name"],
                    "form": item["form"],
                    "changed_entries": item["changed_entries"],
                    "base_sha256": base_hash,
                    "edited_sha256": edited_hash,
                }
            )
            del tensors[item["name"]]
        report[run] = {
            "tensors": len(checked),
            "changed_entries_total": manifest["changed_entries_total"],
            "bit_exact": True,
            "tensors_detail": checked,
        }
        print(f"{run}: {len(checked)} tensors bit-exact", flush=True)
    return {"seconds": round(time.perf_counter() - started, 3), "runs": report}


def _reject_machine_paths(payload: dict) -> None:
    text = json.dumps(payload)
    if "/root/" in text or "/dev/shm" in text:
        raise ValueError("result JSON contains a machine path")


def _write(path: Path, payload: dict) -> None:
    _reject_machine_paths(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1) + "\n")
    temporary.replace(path)


def _last_boxed(text: str):
    marker = "\\boxed"
    start = text.rfind(marker)
    if start < 0:
        return None
    rest = text[start + len(marker) :].lstrip()
    if not rest.startswith("{"):
        return None
    depth = 0
    for index, char in enumerate(rest):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return rest[1:index]
    return None


# mathqa.loop_signals. The last word is dropped because a token cap usually
# stops inside a word. repetitive_end follows the docstring of that function.
# The mcqa.loop_signals helper it calls is not in the public exp8 release.
# The blog cut share is finish_reason "length" at 32,768 tokens, not these flags.
_LOOP_MAX_PERIOD = 200
_LOOP_MIN_REPEATS = 3
_LOOP_MIN_SPAN = 40


def loop_signals(text: str) -> dict:
    words = text.split()
    trimmed = words[:-1]
    count = len(trimmed)
    verbatim = False
    for period in range(1, _LOOP_MAX_PERIOD + 1):
        repeats = max(_LOOP_MIN_REPEATS, -(-_LOOP_MIN_SPAN // period))
        if repeats * period > count:
            break
        block = trimmed[count - period :]
        if all(
            trimmed[count - (offset + 1) * period : count - offset * period] == block
            for offset in range(1, repeats)
        ):
            verbatim = True
            break
    window = words[-200:]
    positions = len(window) - 3
    repetitive = False
    if positions > 0:
        counts: dict[tuple[str, ...], int] = {}
        grams = []
        for index in range(positions):
            gram = tuple(window[index : index + 4])
            grams.append(gram)
            counts[gram] = counts.get(gram, 0) + 1
        hot = sum(1 for gram in grams if counts[gram] >= 3)
        repetitive = hot * 2 >= positions
    return {
        "verbatim_loop_end": verbatim,
        "repetitive_end": repetitive,
        "words": len(words),
    }


def _grade(text: str, gold: str) -> dict:
    pred = _last_boxed(text)
    correct = False
    grader = "no closed boxed answer; math-verify was not called"
    error = None
    if pred is not None:
        grader = "math-verify 0.9.0"
        try:
            import math_verify

            parsed_gold = math_verify.parse("\\boxed{" + gold + "}")
            parsed_pred = math_verify.parse("\\boxed{" + pred + "}")
            correct = bool(
                parsed_gold
                and parsed_pred
                and math_verify.verify(parsed_gold, parsed_pred)
            )
        except Exception as exc:  # noqa: BLE001 - record the grader failure, do not invent a score
            grader = "math-verify-failed"
            error = type(exc).__name__
    return {
        "pred": pred,
        "correct": correct,
        "grader": grader,
        "grader_error": error,
        "saw_end_think": "</think>" in text,
    }


class _OutputPresencePenalty:
    """vLLM output-token presence penalty. The prompt is not penalized."""

    def __init__(self, penalty: float, prompt_length: int):
        self.penalty = penalty
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores):
        import torch

        generated = input_ids[:, self.prompt_length :]
        if generated.numel() == 0:
            return scores
        seen = torch.zeros(scores.shape, dtype=torch.bool, device=scores.device)
        seen.scatter_(1, generated, True)
        return scores - self.penalty * seen.to(scores.dtype)


def _apply_run(model, weights: Path, edits_root: Path, run: str) -> dict:
    import torch

    edit_dir = edits_root / run / "edit"
    manifest = json.loads((edit_dir / "manifest.json").read_text())
    bundle = np.load(edit_dir / "edit.npz")
    names = {item["name"] for item in manifest["tensors"]}
    tensors = _load_named(weights, names)
    changed = 0
    for index, item in enumerate(manifest["tensors"]):
        edited = _rebuild(tensors[item["name"]], item["form"], bundle, index)
        parameter_name = item["name"].replace("model.language_model.", "model.")
        parameter = dict(model.named_parameters())[parameter_name]
        values = torch.from_numpy(_bf16_to_f32(edited)).to(dtype=parameter.dtype)
        if tuple(values.shape) != tuple(parameter.shape):
            raise ValueError(
                f"{parameter_name} shape {values.shape} != {tuple(parameter.shape)}"
            )
        with torch.no_grad():
            parameter.data.copy_(values)
        stored = parameter.detach().to(torch.float32).cpu().numpy()
        if _sha256(_f32_to_bf16_bits(stored)) != item["edited_sha256"]:
            raise ValueError(
                f"{parameter_name} in-memory bits differ from the published edit"
            )
        changed += int(item["changed_entries"])
        del tensors[item["name"]]
    return {"run": run, "changed_entries": changed, "tensors": len(manifest["tensors"])}


def generate_smoke(args) -> dict:
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        LogitsProcessorList,
    )

    torch.set_num_threads(2)
    problem = json.loads(Path(args.problem_json).read_text())
    config = AutoConfig.from_pretrained(args.text_config)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    template_path = Path(args.text_config) / "chat_template.jinja"
    if template_path.is_file():
        tokenizer.chat_template = template_path.read_text()
    user = problem["problem"].rstrip() + "\n" + INSTRUCTION
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if not prompt.endswith("<think>\n"):
        raise ValueError("thinking prompt did not end with an open think tag")
    inputs = tokenizer(prompt, return_tensors="pt")
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.tokenizer,
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_seconds = time.perf_counter() - started
    import importlib.util

    kernel_fallback = [
        name
        for name, module in (
            ("causal_conv1d", "causal_conv1d"),
            ("flash-linear-attention", "fla"),
        )
        if importlib.util.find_spec(module) is None
    ]
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"loaded {type(model).__name__} params={parameters} in {load_seconds:.1f}s",
        flush=True,
    )

    def one_sample(label: str) -> dict:
        torch.manual_seed(PROTOCOL_SAMPLING["seed"])
        processor = LogitsProcessorList(
            [
                _OutputPresencePenalty(
                    PROTOCOL_SAMPLING["presence_penalty"], inputs["input_ids"].shape[1]
                )
            ]
        )
        begun = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                do_sample=True,
                temperature=PROTOCOL_SAMPLING["temperature"],
                top_p=PROTOCOL_SAMPLING["top_p"],
                top_k=PROTOCOL_SAMPLING["top_k"],
                max_new_tokens=args.max_new_tokens,
                logits_processor=processor,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - begun
        new_ids = output[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(new_ids, skip_special_tokens=False)
        graded = _grade(text, problem["answer"])
        signals = loop_signals(text)
        hit_cap = (
            int(new_ids.shape[0]) >= args.max_new_tokens
            and int(new_ids[-1]) != tokenizer.eos_token_id
        )
        row = {
            "label": label,
            "new_tokens": int(new_ids.shape[0]),
            "seconds": round(elapsed, 3),
            "tokens_per_second": round(int(new_ids.shape[0]) / elapsed, 4)
            if elapsed
            else None,
            "hit_length_cap": bool(hit_cap),
            "verbatim_loop_end": signals["verbatim_loop_end"],
            "repetitive_end": signals["repetitive_end"],
            "words": signals["words"],
            "text": text,
            **graded,
        }
        print(
            f"{label}: {row['new_tokens']} tokens in {row['seconds']}s "
            f"cap={row['hit_length_cap']} boxed={row['pred']!r} "
            f"verbatim_loop={row['verbatim_loop_end']} repetitive={row['repetitive_end']}",
            flush=True,
        )
        return row

    samples = [one_sample("untouched")]
    edit = None
    if args.edit_run:
        edit = _apply_run(
            model, Path(args.weights), Path(args.edits_root), args.edit_run
        )
        samples.append(one_sample(args.edit_run))
    return {
        "load_seconds": round(load_seconds, 3),
        "architecture": type(model).__name__,
        "parameters": parameters,
        "dtype": "bfloat16",
        "device": "cpu",
        "kernel_fallback": kernel_fallback,
        "problem_id": problem["unique_id"],
        "subject": problem.get("subject"),
        "level": problem.get("level"),
        "gold": problem["answer"],
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "max_new_tokens": args.max_new_tokens,
        "protocol_max_new_tokens": PROTOCOL_SAMPLING["max_new_tokens"],
        "samples_per_problem": 1,
        "problems": 1,
        "published_math500_samples": 4,
        "published_math500_problems": 500,
        "edit_applied": edit,
        "loop_metrics": (
            "verbatim_loop_end matches mathqa.loop_signals, which drops the last word. "
            "repetitive_end follows that docstring; mcqa.loop_signals is not in the public release. "
            "The blog cut share counts finish_reason length at 32768 tokens."
        ),
        "samples": samples,
    }


def build_report(
    args, edits: dict, smoke: dict | None, smoke_error: str | None
) -> dict:
    return {
        "status": "not_reproduced",
        "kind": "published_edit_bitexact_and_bounded_smoke"
        if smoke
        else "published_edit_bitexact",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "isolated_transformers": "5.17.0",
        "product_transformers_unchanged": "4.57.1",
        "published_scores_not_rerun": {
            "source": "exp8_math/eval/*/merged.json",
            "unedited_math500_acc": 0.079,
            "unedited_math500_cut_share": 0.93,
            "unedited_gsm8k_acc": 0.13836239575435935,
            "p10c_math500_acc": 0.1235,
            "p25c_math500_acc": 0.141,
            "p50c_math500_acc": 0.481,
            "p50c_gsm8k_acc": 0.6709628506444276,
            "p50c_math500_cut_share": 0.3305,
        },
        "edits": edits,
        "smoke": smoke,
        "smoke_error": smoke_error,
        "deviations": [
            "The product runtime stays on transformers 4.57.1 and still fails closed for qwen3_5. Loading used an isolated transformers 5.17.0.",
            "Generation, when run, uses Hugging Face on CPU. The published evaluator uses vLLM.",
            "transformers 5.17.0 has no presence_penalty generation argument. The smoke subtracts 1.5 from a logit when that token already occurred in the generated span, which is vLLM's output-token presence penalty, and then applies temperature, top-k, and top-p.",
            "The smoke length cap is not 32768, and the smoke is one problem and one sample. It is not the 500-problem four-sample MATH-500 score or the 93 percent unedited cut rate.",
            "No frontier editor ran. Astra was gpt-6-astra through Codex. A Grok editor was not called. The direct edit, when applied, is the published diff.",
            "The public checkpoint is multimodal. The smoke loads the published text config (qwen3_5_text, Qwen3_5ForCausalLM) and does not load the vision tower. Edit tensor names in the file remain model.language_model.*.",
            "The harness driver, forward-pass server, and practice activations are not in the public logs. The prompts that were sent, the edit diffs, the score files, and the evaluation code are.",
            "causal_conv1d and flash-linear-attention are not installed. Transformers used its reference PyTorch kernels for the convolution and gated delta rule.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--edits-root", required=True)
    parser.add_argument("--text-config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--problem-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--edit-run", choices=("p10c", "p25c", "p50c"))
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    edits = verify_edits(Path(args.weights), Path(args.edits_root))
    report = build_report(args, edits, None, None)
    output = Path(args.output)
    _write(output, report)
    if not args.generate:
        print(f"wrote {output.name}", flush=True)
        return
    try:
        smoke = generate_smoke(args)
    except Exception as exc:  # noqa: BLE001 - persist the edit result, then surface the generation failure
        report["smoke_error"] = f"{type(exc).__name__}: {exc}"
        _write(output, report)
        raise
    report = build_report(args, edits, smoke, None)
    _write(output, report)
    print(f"wrote {output.name}", flush=True)


if __name__ == "__main__":
    main()
