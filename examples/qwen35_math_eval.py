"""Execute Qwen3.5 under the math benchmark harness.

This file loads the model, applies one published edit, and returns generations.
Grading, resume, loop metrics, and the equal-budget comparison stay in
``weightscript.math_benchmark``. A token cap below 32,768, one sample, a local
JSONL split, and a Transformers CPU engine are recorded deviations. They do
not reproduce the blog table.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

from weightscript.math_benchmark import (
    Budgets,
    QWEN35_BASE_MODEL,
    QWEN35_BASE_REVISION,
    RunSpec,
    compare_runs,
    load_jsonl,
    run_evaluation,
    write_summary,
)

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from qwen35_blog_protocol import (  # noqa: E402
    _OutputPresencePenalty,
    _apply_run,
    _reject_machine_paths,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _edited_fingerprint(base_sha: str, edits_root: Path, run: str) -> str:
    manifest = json.loads((edits_root / run / "edit" / "manifest.json").read_text())
    payload = {
        "base_sha256": base_sha,
        "run": run,
        "tensors": [
            {"edited_sha256": item["edited_sha256"], "name": item["name"]}
            for item in manifest["tensors"]
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _generate(model, tokenizer):
    import torch
    from transformers import LogitsProcessorList

    def generate(requests):
        responses = []
        for request in requests:
            sampling = request["sampling"]
            prompt = request["rendered_prompt"]
            if not isinstance(prompt, str) or not prompt.endswith("<think>\n"):
                raise RuntimeError("thinking prompt did not end with an open think tag")
            required = (
                "temperature",
                "top_p",
                "top_k",
                "presence_penalty",
                "max_tokens",
            )
            missing = [key for key in required if key not in sampling]
            if missing:
                raise RuntimeError(f"sampling is missing {missing}")
            inputs = tokenizer(prompt, return_tensors="pt")
            torch.manual_seed(int(request["seed"]))
            processor = LogitsProcessorList(
                [
                    _OutputPresencePenalty(
                        float(sampling["presence_penalty"]),
                        inputs["input_ids"].shape[1],
                    )
                ]
            )
            begun = time.perf_counter()
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=float(sampling["temperature"]),
                    top_p=float(sampling["top_p"]),
                    top_k=int(sampling["top_k"]),
                    max_new_tokens=int(sampling["max_tokens"]),
                    logits_processor=processor,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            elapsed = time.perf_counter() - begun
            new_ids = output[0, inputs["input_ids"].shape[1] :]
            count = int(new_ids.shape[0])
            hit_length = (
                count >= int(sampling["max_tokens"])
                and int(new_ids[-1]) != tokenizer.eos_token_id
            )
            text = tokenizer.decode(new_ids, skip_special_tokens=False)
            print(
                f"{request['example_id']} sample {request['sample_index']}: "
                f"{count} tokens in {elapsed:.3f}s finish="
                f"{'length' if hit_length else 'stop'}",
                flush=True,
            )
            responses.append(
                {
                    "example_id": request["example_id"],
                    "text": text,
                    "finish_reason": "length" if hit_length else "stop",
                    "n_tokens": count,
                }
            )
        return responses

    return generate


def _load(args):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(2)
    config = AutoConfig.from_pretrained(args.text_config)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    template = Path(args.text_config) / "chat_template.jinja"
    if template.is_file():
        tokenizer.chat_template = template.read_text()
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.tokenizer,
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    fallback = [
        name
        for name, module in (
            ("causal_conv1d", "causal_conv1d"),
            ("flash-linear-attention", "fla"),
        )
        if importlib.util.find_spec(module) is None
    ]
    parameters = sum(parameter.numel() for parameter in model.parameters())
    load_seconds = time.perf_counter() - started
    print(
        f"loaded {type(model).__name__} params={parameters} in {load_seconds:.1f}s",
        flush=True,
    )
    return (
        model,
        tokenizer,
        {
            "architecture": type(model).__name__,
            "parameters": parameters,
            "dtype": "bfloat16",
            "device": "cpu",
            "kernel_fallback": fallback,
            "load_seconds": round(load_seconds, 3),
        },
    )


def _assert_public(payload: dict) -> None:
    _reject_machine_paths(payload)
    if "/mnt/" in json.dumps(payload):
        raise ValueError("result JSON contains a machine path")


def _assert_public_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "/root/" in text or "/dev/shm" in text or "/mnt/" in text:
        raise ValueError(f"{path.name} contains a machine path")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--edits-root", required=True)
    parser.add_argument("--text-config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--edit-run", default="p50c")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    examples = load_jsonl(
        args.dataset, dataset="math500", split="test", role="evaluation"
    )
    budgets = Budgets(
        max_examples=args.max_examples,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
    )
    base_sha = _file_sha256(Path(args.weights))
    model, tokenizer, loaded = _load(args)
    generate = _generate(model, tokenizer)
    common = {
        "source_model": QWEN35_BASE_MODEL,
        "source_revision": QWEN35_BASE_REVISION,
        "protocol_name": "think35",
        "engine": "transformers",
        "dtype": "bfloat16",
    }
    untouched_path = output / "untouched.jsonl"
    edited_path = output / "edited.jsonl"
    run_evaluation(
        examples,
        RunSpec(
            checkpoint_id="untouched",
            checkpoint_fingerprint=base_sha,
            arm="untouched",
            **common,
        ),
        generate,
        untouched_path,
        budgets=budgets,
        tokenizer=tokenizer,
    )
    edit = _apply_run(model, Path(args.weights), Path(args.edits_root), args.edit_run)
    run_evaluation(
        examples,
        RunSpec(
            checkpoint_id=args.edit_run,
            checkpoint_fingerprint=_edited_fingerprint(
                base_sha, Path(args.edits_root), args.edit_run
            ),
            arm="edited",
            **common,
        ),
        generate,
        edited_path,
        budgets=budgets,
        tokenizer=tokenizer,
    )
    report = compare_runs(untouched_path, edited_path)
    _assert_public(report)
    compare_path = output / "compare.json"
    write_summary(compare_path, report)
    status = {
        "status": "not_reproduced",
        "kind": "harness_prefix_transformers_cpu",
        "model_id": QWEN35_BASE_MODEL,
        "revision": QWEN35_BASE_REVISION,
        "protocol": "think35",
        "dataset": "math500",
        "split": "test",
        "selection": "canonical_prefix",
        "max_examples": args.max_examples,
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "published_max_new_tokens": 32768,
        "published_math500_samples": 4,
        "published_math500_problems": 500,
        "engine": "transformers",
        "edit_applied": edit,
        "load": loaded,
        "blog_table_settings": report["blog_table_settings"],
        "comparable": report["comparable"],
        "complete": report["complete"],
        "blog_table_blockers": report["blog_table_blockers"],
        "incomparable_because": report.get("incomparable_because"),
        "untouched_summary": report["untouched"],
        "edited_summary": report["edited"],
    }
    _assert_public(status)
    status_path = output / "status.json"
    status_path.write_text(json.dumps(status, indent=1, sort_keys=True) + "\n")
    for path in (untouched_path, edited_path, compare_path, status_path):
        _assert_public_file(path)
    print(
        "blog_table_settings="
        f"{report['blog_table_settings']} comparable={report['comparable']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
