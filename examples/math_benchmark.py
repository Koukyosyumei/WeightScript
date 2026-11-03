"""MATH-500 / GSM8K harness for a weight-space programming reproduction.

This does not download a model and does not run inference. Main supplies a
generator. See docs/math-benchmark.md.

    PYTHONPATH=python python examples/math_benchmark.py protocol
    PYTHONPATH=python python examples/math_benchmark.py smoke --out /tmp/math-bench-smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from weightscript.math_benchmark import (  # noqa: E402
    Budgets,
    GenerationRejected,
    RunSpec,
    compare_runs,
    dataset_cache_dir,
    fetch_split,
    load_jsonl,
    protocol,
    run_evaluation,
    write_summary,
)


def _spec(args, arm: str, checkpoint: str, fingerprint: str) -> RunSpec:
    return RunSpec(
        source_model=args.source_model,
        source_revision=args.source_revision,
        checkpoint_id=checkpoint,
        checkpoint_fingerprint=fingerprint,
        arm=arm,
        protocol_name=args.protocol,
        engine=args.engine,
        dtype=args.dtype,
    )


def _budgets(args) -> Budgets:
    return Budgets(
        max_examples=args.max_examples,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        walltime_s=args.walltime_s,
    )


def _tokenizer(name: str | None):
    if name is None:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def _print_summary(summary: dict) -> None:
    fields = (
        "arm",
        "dataset",
        "n_problems",
        "n_configured",
        "samples",
        "complete",
        "acc",
        "acc_equiv",
        "acc_lenient",
        "headline",
        "headline_value",
        "cut_share",
        "mean_tokens",
        "verbatim_loop_end_share",
        "repetitive_end_share",
        "stopped_reason",
        "deviations",
        "metrics_are_article_grader",
    )
    print(json.dumps({key: summary[key] for key in fields}, indent=1, ensure_ascii=False))


def command_protocol(args) -> int:
    item = protocol(args.protocol)
    print(
        json.dumps(
            {
                "name": item.name,
                "thinking": item.thinking,
                "sampling": item.sampling,
                "samples": item.samples,
                "instruction": item.instruction,
                "headline": item.headline,
                "engine": item.engine,
                "dtype": item.dtype,
                "source": item.source,
            },
            indent=1,
            ensure_ascii=False,
        )
    )
    return 0


def command_run(args) -> int:
    examples = load_jsonl(args.dataset, dataset=args.dataset_name, split=args.split, role="evaluation")
    generations = None
    if args.generations:
        generations = {}
        for line in Path(args.generations).read_text().splitlines():
            if line.strip() == "":
                continue
            row = json.loads(line)
            generations[(row["example_id"], row["sample_index"])] = row

    def generate(requests):
        if generations is None:
            raise SystemExit("run needs --generations; a live model generator belongs in the caller's process")
        found = []
        for request in requests:
            if "gold" in request or "answer" in request:
                raise SystemExit("harness request leaked an answer")
            key = (request["example_id"], request["sample_index"])
            try:
                found.append(generations[key])
            except KeyError as error:
                raise GenerationRejected(f"no stored generation for {key}") from error
        return found

    summary = run_evaluation(
        examples,
        _spec(args, args.arm, args.checkpoint_id, args.checkpoint_fingerprint),
        generate,
        args.out,
        budgets=_budgets(args),
        tokenizer=_tokenizer(args.tokenizer),
    )
    write_summary(Path(args.out).with_suffix(Path(args.out).suffix + ".summary.json"), summary)
    _print_summary(summary)
    print("blog numbers require compare's blog_table_settings; this summary does not assert them")
    return 0


def command_compare(args) -> int:
    report = compare_runs(args.untouched, args.edited)
    printable = {key: value for key, value in report.items() if key != "per_example"}
    printable["per_example_count"] = None if report["per_example"] is None else len(report["per_example"])
    print(json.dumps(printable, indent=1, ensure_ascii=False))
    if args.per_example and report["per_example"] is not None:
        destination = Path(args.per_example)
        destination.write_text(json.dumps(report["per_example"], indent=1, ensure_ascii=False) + "\n")
    if not report["comparable"] or not report["complete"]:
        return 2
    return 0


def command_smoke(args) -> int:
    """Canned generations. Not a MATH-500 or GSM8K score."""

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    problems = out / "problems.jsonl"
    problems.write_text(
        "\n".join(
            [
                json.dumps({"id": "p1", "problem": "What is 20+4?", "answer": "24"}),
                json.dumps({"id": "p2", "problem": "What is 3+4?", "answer": "7"}),
            ]
        )
        + "\n"
    )
    examples = load_jsonl(problems, dataset="fixture", split="test", role="evaluation")
    calls = {"n": 0}

    def untouched(requests):
        calls["n"] += 1
        for request in requests:
            if {"gold", "answer", "solution"} & set(request):
                raise SystemExit("request leaked an answer")
        return [
            {"text": "\\boxed{0}", "finish_reason": "stop", "n_tokens": 3}
            for _request in requests
        ]

    def edited(requests):
        canned = {"p1": "24", "p2": "7"}
        return [
            {
                "text": "\\boxed{" + canned[request["example_id"]] + "}",
                "finish_reason": "length" if request["example_id"] == "p2" else "stop",
                "n_tokens": 5,
            }
            for request in requests
        ]

    common = dict(
        source_model=args.source_model,
        source_revision=args.source_revision,
        protocol_name=args.protocol,
        engine=args.engine,
        dtype=args.dtype,
    )
    budgets = Budgets(max_samples=1, max_new_tokens=32, walltime_s=args.walltime_s)
    first = run_evaluation(
        examples,
        RunSpec(checkpoint_id="untouched-smoke", checkpoint_fingerprint="fp-untouched", arm="untouched", **common),
        untouched,
        out / "untouched.jsonl",
        budgets=budgets,
    )
    run_evaluation(
        examples,
        RunSpec(checkpoint_id="untouched-smoke", checkpoint_fingerprint="fp-untouched", arm="untouched", **common),
        untouched,
        out / "untouched.jsonl",
        budgets=budgets,
    )
    second = run_evaluation(
        examples,
        RunSpec(checkpoint_id="edited-smoke", checkpoint_fingerprint="fp-edited", arm="edited", **common),
        edited,
        out / "edited.jsonl",
        budgets=budgets,
    )
    report = compare_runs(out / "untouched.jsonl", out / "edited.jsonl")
    print(
        json.dumps(
            {
                "note": "Smoke used canned text. It is not a blog-table score.",
                "generator_calls_for_untouched": calls["n"],
                "untouched_acc": first["acc"],
                "untouched_acc_equiv": first["acc_equiv"],
                "edited_acc": second["acc"],
                "edited_cut_share": second["cut_share"],
                "comparable": report["comparable"],
                "complete": report["complete"],
                "delta_acc": report["delta_acc"],
                "blog_table_settings": report["blog_table_settings"],
                "blog_table_blockers": report["blog_table_blockers"],
            },
            indent=1,
            ensure_ascii=False,
        )
    )
    write_summary(out / "compare.json", {key: value for key, value in report.items() if key != "per_example"})
    return 0 if report["comparable"] and report["complete"] else 2


def command_fetch(args) -> int:
    dest = args.dest or dataset_cache_dir()
    if dest is None:
        raise SystemExit("pass --dest or set WEIGHTSCRIPT_CACHE_DIR")
    report = fetch_split(args.dataset_name, args.split, dest)
    print(json.dumps(report, indent=1, ensure_ascii=False, sort_keys=True))
    return 0 if report.get("count_matches_published") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Weight-space programming math evaluation harness")
    parser.add_argument("--protocol", default="think35")
    parser.add_argument("--source-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--source-revision", default="2fc06364715b967f1860aea9cf38778875588b17")
    parser.add_argument("--engine", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--walltime-s", type=float, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    protocol_parser = sub.add_parser("protocol", help="print one published protocol")
    protocol_parser.set_defaults(func=command_protocol)

    run_parser = sub.add_parser("run", help="grade stored generations; does not call a model")
    run_parser.add_argument("--dataset", required=True)
    run_parser.add_argument("--dataset-name", required=True, choices=("math500", "gsm8k", "fixture"))
    run_parser.add_argument("--split", required=True)
    run_parser.add_argument("--generations", required=True, help="JSONL of example_id, sample_index, text, finish_reason")
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--arm", required=True, choices=("untouched", "edited"))
    run_parser.add_argument("--checkpoint-id", required=True)
    run_parser.add_argument("--checkpoint-fingerprint", required=True)
    run_parser.add_argument("--tokenizer", default=None, help="optional local tokenizer; applies enable_thinking")
    run_parser.set_defaults(func=command_run)

    compare_parser = sub.add_parser("compare", help="compare untouched and edited result files")
    compare_parser.add_argument("--untouched", required=True)
    compare_parser.add_argument("--edited", required=True)
    compare_parser.add_argument("--per-example", default=None)
    compare_parser.set_defaults(func=command_compare)

    smoke_parser = sub.add_parser("smoke", help="canned two-problem run; not a benchmark score")
    smoke_parser.add_argument("--out", required=True)
    smoke_parser.set_defaults(func=command_smoke)

    fetch_parser = sub.add_parser("fetch", help="download one published split into a chosen directory")
    fetch_parser.add_argument("--dataset-name", required=True, choices=("math500", "gsm8k"))
    fetch_parser.add_argument("--split", required=True)
    fetch_parser.add_argument("--dest", default=None, help="directory; or set WEIGHTSCRIPT_CACHE_DIR")
    fetch_parser.set_defaults(func=command_fetch)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
