"""Learned weights -> upstream pruned circuit -> matched rule -> edited weights.

Two lowering paths: Tracr creates a new model; constraint compilation patches
an existing trained GPT-2. Every approximation and bounded domain is recorded.
"""

import argparse
import itertools
import json
from pathlib import Path

import torch
from weightscript.backends.tracr import compile_program
from weightscript.backends.upstream import decompile
from weightscript.backends.upstream_stages import decompile_stages
from weightscript.checkpoint import save_undo, undo
from weightscript.core import audit, fingerprint, load_model, write_json
from weightscript.symbolic import lower, synthesize_binary
from weightscript.synthesis import cegis


def domain(lengths):
    return [
        {"input_ids": [2] + list(bits) + [3]}
        for n in lengths
        for bits in itertools.product((0, 1), repeat=n)
    ]


def _contents(lengths, per_length, seed, population, forbidden):
    rng = __import__("random").Random(seed)
    chosen = []
    used = set(forbidden)
    attempts = 0
    for length in lengths:
        made = 0
        while made < per_length:
            attempts += 1
            if attempts > 100000:
                raise RuntimeError("could not sample a disjoint content domain")
            content = tuple(rng.sample(range(population), length))
            if content in used:
                continue
            used.add(content)
            chosen.append(content)
            made += 1
    return chosen, used


def _pack(contents, bos, sep, continuation):
    return [
        {"input_ids": [bos, *content, sep, *continuation(content)]}
        for content in contents
    ]


def _prefixes(rows, sep):
    return [
        {"input_ids": row["input_ids"][: row["input_ids"].index(sep) + 1]}
        for row in rows
    ]


def _native_task(model, rows, sep):
    from weightscript.backends.rules import supervised_pairs
    from weightscript.evaluate import batches

    correct = total = sequences = 0
    for chunk, ids in batches(rows, 32):
        logits = model(ids, use_cache=False).logits
        for row, row_logits in zip(chunk, logits):
            pairs = supervised_pairs(row["input_ids"], sep)
            good = True
            for prefix, gold in pairs:
                match = int(row_logits[len(prefix) - 1].argmax()) == gold
                correct += match
                total += 1
                good = good and match
            sequences += int(good)
    return {
        "position_agreement": correct / total,
        "sequence_agreement": sequences / len(rows),
        "positions": total,
        "sequences": len(rows),
    }


def _replay_task(model, artifact, checkout, rows, sep, stage):
    from weightscript.backends.rules import supervised_pairs
    from weightscript.backends.upstream import execute_artifact
    from weightscript.backends.upstream_stages import execute_stages
    from weightscript.evaluate import batches

    stage_correct = native_correct = total = 0
    failures = []
    for chunk, ids in batches(rows, 16):
        native = model(ids, use_cache=False).logits
        if stage == "component":
            nested = next(
                item
                for item in artifact["nodes"][-1]["attributes"]["stages"]
                if item["name"] == "component"
            )["artifact"]
            actual = execute_artifact(model, nested, checkout, ids)
        else:
            actual = execute_stages(model, artifact, checkout, ids, stage)
        for row, native_row, actual_row in zip(chunk, native, actual):
            for prefix, gold in supervised_pairs(row["input_ids"], sep):
                index = len(prefix) - 1
                native_token = int(native_row[index].argmax())
                stage_token = int(actual_row[index].argmax())
                native_correct += native_token == gold
                stage_correct += stage_token == native_token
                total += 1
                if stage_token != native_token and len(failures) < 8:
                    failures.append(
                        {
                            "input_ids": prefix,
                            "stage": stage_token,
                            "native": native_token,
                            "gold": gold,
                        }
                    )
    return {
        "stage_vs_native": stage_correct / total,
        "native_vs_gold": native_correct / total,
        "positions": total,
        "counterexamples": failures,
    }


def _edit(model, rule, calibration, oracle, final):
    from weightscript.core import fingerprint
    from weightscript.synthesis import cegis

    source = fingerprint(model)
    try:
        _edited, report = cegis(
            model, rule, calibration, oracle, heldout=final, max_rounds=3
        )
    except ValueError as exc:
        report = {"bounded_oracle_passed": False, "error": str(exc)}
    if fingerprint(model) != source:
        raise RuntimeError("semantic edit mutated source weights")
    history = report.pop("history", None)
    if history:
        report["rounds_slim"] = [
            {
                "passed": item["passed"],
                "synthesis": item["synthesis"],
                "verification": item["verification"],
                "patch_rank_bound": item["patch_rank_bound"],
                "solver": item["solver"],
                "changed_scalars": item["audit"]["changed_scalars"],
            }
            for item in history
        ]
    return report


def run_suite(checkout, output, tasks):
    """Execute trained majority/copy/sort cases and write measured JSON reports."""
    import traceback

    from weightscript.backends.pipeline import disjoint_domains, summarize_stages
    from weightscript.backends.upstream_stages import decompile_stages
    from weightscript.core import fingerprint, load_model, write_json
    from weightscript.symbolic import synthesize_sequence_rules

    names = {
        "majority": "bin_majority-@1l1h16d3lr01drop",
        "copy": "unique_copy-@2l1h64d3lr01drop",
        "sort": "sort-@1l1h256d3lr01drop",
    }
    specs = {
        "majority": {
            "bos": 2,
            "sep": 3,
            "eos": 4,
            "population": 2,
            "kind": "majority",
            "calibration": (1, 3, 5),
            "oracle": (7,),
            "final": (9,),
            "per": (1, 1, 1),
        },
        "copy": {
            "bos": 150,
            "sep": 151,
            "eos": 152,
            "population": 150,
            "kind": "copy",
            "calibration": (2, 3, 4),
            "oracle": (6,),
            "final": (8,),
            "per": (8, 16, 16),
        },
        "sort": {
            "bos": 150,
            "sep": 151,
            "eos": 152,
            "population": 150,
            "kind": "sorted",
            "calibration": (2, 3, 4),
            "oracle": (6,),
            "final": (8,),
            "per": (8, 16, 16),
        },
    }
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for task in tasks:
        spec = specs[task]
        bos, sep, eos = spec["bos"], spec["sep"], spec["eos"]
        if task == "majority":

            def bits(lengths):
                chosen = []
                for length in lengths:
                    for item in itertools.product((0, 1), repeat=length):
                        if item.count(0) != item.count(1):
                            chosen.append(item)
                return chosen

            groups = [
                bits(spec["calibration"]),
                bits(spec["oracle"]),
                bits(spec["final"]),
            ]
        else:
            used = set()
            groups = []
            for lengths, count, seed in zip(
                (spec["calibration"], spec["oracle"], spec["final"]),
                spec["per"],
                (11, 22, 33),
            ):
                chosen, used = _contents(lengths, count, seed, spec["population"], used)
                groups.append(chosen)

        def continuation(content, kind=spec["kind"], eos_id=eos):
            if kind == "majority":
                return [0 if content.count(0) > content.count(1) else 1]
            if kind == "copy":
                return [*content, eos_id]
            return [*sorted(content), eos_id]

        full = [_pack(group, bos, sep, continuation) for group in groups]
        prefixes = [_prefixes(rows, sep) for rows in full]
        domain_hashes = disjoint_domains(
            calibration=prefixes[0], oracle=prefixes[1], final=prefixes[2]
        )
        disjoint_domains(calibration=full[0], oracle=full[1], final=full[2])
        model = load_model(str(Path(checkout) / "share/saved_models" / names[task]))
        source = fingerprint(model)
        markers = {"bos": bos, "sep": sep, "eos": eos}
        candidates = [
            {"op": "span_next", "kind": kind, **markers}
            for kind in ("copy", "sorted", "majority")
        ]
        semantic = synthesize_sequence_rules(
            model, candidates, full[0], full[1], full[2], min_fidelity=0.99
        )
        if fingerprint(model) != source:
            raise RuntimeError("sequence synthesis mutated source weights")
        case = {
            "checkpoint": names[task],
            "source_sha256": source,
            "task_kind": spec["kind"],
            "domains": domain_hashes,
            "domain_sizes": {
                "calibration": len(full[0]),
                "oracle": len(full[1]),
                "final": len(full[2]),
            },
            "position_assumption": "implicit positions 0..length-1; upstream training also drew random offsets",
            "native_task": {
                name: _native_task(model, rows, sep)
                for name, rows in zip(("calibration", "oracle", "final"), full)
            },
            "semantic": semantic.to_dict(),
        }
        artifact = None
        try:
            causal = decompile_stages(
                model,
                checkout,
                prefixes[0],
                prefixes[1],
                max_trials=64,
                calibration_steps=32,
                min_fidelity=0.99,
                batch_size=32,
            )
            if fingerprint(model) != source:
                raise RuntimeError("stage driver mutated source weights")
            artifact = causal.to_dict()
            case["causal_summary"] = summarize_stages(artifact)
            case["replay"] = {}
            for stage in ("component", "full_path", "qk"):
                case["replay"][stage] = {
                    name: _replay_task(model, artifact, checkout, rows, sep, stage)
                    for name, rows in (("oracle", full[1]), ("final", full[2]))
                }
            if task == "majority":
                edit_rule = {"op": "count_threshold", "threshold": 0.5, "invert": True}
                case["edit"] = _edit(
                    model, edit_rule, prefixes[0], prefixes[1], prefixes[2]
                )
                case["edit"]["requested_rule"] = edit_rule
                case["edit"]["limit"] = (
                    "last-position output-span patch of the trained checkpoint; held-out rows were not used to fit it"
                )
            else:
                target = {
                    "op": "span_next",
                    "kind": "sorted" if task == "copy" else "copy",
                    **markers,
                }
                from weightscript.backends.rules import supervised_pairs

                # The compiler supervises only the last position of each prefix.
                bounded = {"calibration": [], "oracle": [], "final": []}
                for name, rows in zip(("calibration", "oracle", "final"), full):
                    for row in rows[:4]:
                        prefix, _gold = supervised_pairs(row["input_ids"], sep)[0]
                        bounded[name].append({"input_ids": prefix})
                case["edit"] = _edit(
                    model,
                    target,
                    bounded["calibration"],
                    bounded["oracle"],
                    bounded["final"],
                )
                case["edit"]["requested_rule"] = target
                case["edit"]["examples"] = [
                    len(bounded[name]) for name in ("calibration", "oracle", "final")
                ]
                case["edit"]["limit"] = (
                    "bounded last-position patch on four prefixes; not a sequence-wide program transplant"
                )
            if fingerprint(model) != source:
                raise RuntimeError("trained case mutated source weights")
        except Exception:  # noqa: BLE001 - persist this task, then continue the suite
            case["task_error"] = traceback.format_exc()
        kept = {key: case[key] for key in case if key != "semantic"}
        kept["semantic_fidelity"] = semantic.fidelity
        kept["semantic_rule"] = (
            None
            if semantic.output != "hypothesis"
            else semantic.nodes[-1].attributes["rule"]
        )
        kept["semantic_method"] = semantic.provenance["method"]
        summary[task] = kept
        write_json(out / f"{task}-summary.json", kept)
        if artifact is not None:
            write_json(out / f"{task}-stages.json", artifact)
        print(
            task,
            json.dumps(
                {
                    "causal_accepted": None
                    if "causal_summary" not in case
                    else case["causal_summary"]["accepted"],
                    "semantic": semantic.fidelity["accepted"],
                    "native_final": case["native_task"]["final"],
                    "task_error": "task_error" in case,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    write_json(
        out / "suite-index.json",
        {
            task: {
                "checkpoint": summary[task]["checkpoint"],
                "source_sha256": summary[task]["source_sha256"],
            }
            for task in summary
        },
    )
    return summary


def _with_answers(rows, sep):
    answered = []
    for row in rows:
        ids = row["input_ids"]
        start = ids.index(sep) + 1
        if start >= len(ids):
            raise ValueError("sequence has no answer token after the separator")
        answered.append({"input_ids": ids, "answer_start": start})
    return answered


def _replay_coefficient(model, artifact, checkout, rows):
    from weightscript.backends.upstream import (
        agreement_metrics,
        empty_decision_counts,
        score_decisions,
    )
    from weightscript.backends.upstream_search import execute_coefficient
    from weightscript.evaluate import batches

    counts = empty_decision_counts()
    for chunk, ids in batches(rows, 8):
        native = model(ids, use_cache=False).logits
        actual = execute_coefficient(
            model, artifact, checkout, ids, "qk", apply_mlp=False
        )
        score_decisions(chunk, actual, native, counts)
    return agreement_metrics(counts)


def run_coefficient(checkout, output, tasks, steps, pretrain_steps, max_seconds):
    """Bounded coefficient-search pilot. Does not overwrite the greedy rejection files."""
    import traceback

    from weightscript.backends.pipeline import disjoint_domains, summarize_stages
    from weightscript.backends.upstream_search import decompile_coefficient
    from weightscript.core import fingerprint, load_model, write_json

    names = {
        "majority": "bin_majority-@1l1h16d3lr01drop",
        "copy": "unique_copy-@2l1h64d3lr01drop",
        "sort": "sort-@1l1h256d3lr01drop",
    }
    specs = {
        "majority": {
            "bos": 2,
            "sep": 3,
            "eos": 4,
            "population": 2,
            "kind": "majority",
            "calibration": (1, 3, 5),
            "oracle": (7,),
            "final": (9,),
            "per": (1, 1, 1),
        },
        "copy": {
            "bos": 150,
            "sep": 151,
            "eos": 152,
            "population": 150,
            "kind": "copy",
            "calibration": (2, 3, 4),
            "oracle": (6,),
            "final": (8,),
            "per": (8, 16, 16),
        },
        "sort": {
            "bos": 150,
            "sep": 151,
            "eos": 152,
            "population": 150,
            "kind": "sorted",
            "calibration": (2, 3, 4),
            "oracle": (6,),
            "final": (8,),
            "per": (8, 16, 16),
        },
    }
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for task in tasks:
        spec = specs[task]
        bos, sep, eos = spec["bos"], spec["sep"], spec["eos"]
        if task == "majority":

            def bits(lengths):
                chosen = []
                for length in lengths:
                    for item in itertools.product((0, 1), repeat=length):
                        if item.count(0) != item.count(1):
                            chosen.append(item)
                return chosen

            groups = [
                bits(spec["calibration"]),
                bits(spec["oracle"]),
                bits(spec["final"]),
            ]
        else:
            used = set()
            groups = []
            for lengths, count, seed in zip(
                (spec["calibration"], spec["oracle"], spec["final"]),
                spec["per"],
                (11, 22, 33),
            ):
                chosen, used = _contents(lengths, count, seed, spec["population"], used)
                groups.append(chosen)

        def continuation(content, kind=spec["kind"], eos_id=eos):
            if kind == "majority":
                return [0 if content.count(0) > content.count(1) else 1]
            if kind == "copy":
                return [*content, eos_id]
            return [*sorted(content), eos_id]

        full = [
            _with_answers(_pack(group, bos, sep, continuation), sep) for group in groups
        ]
        domain_hashes = disjoint_domains(
            calibration=full[0], oracle=full[1], final=full[2]
        )
        print(
            f"coefficient {task} steps={steps} pretrain={pretrain_steps} cap={max_seconds}s",
            flush=True,
        )
        model = load_model(str(Path(checkout) / "share/saved_models" / names[task]))
        source = fingerprint(model)
        case = {
            "checkpoint": names[task],
            "source_sha256": source,
            "task_kind": spec["kind"],
            "domains": domain_hashes,
            "domain_sizes": {
                name: len(rows)
                for name, rows in zip(("calibration", "oracle", "final"), full)
            },
            "answer_positions": {
                name: sum(len(row["input_ids"]) - row["answer_start"] for row in rows)
                for name, rows in zip(("calibration", "oracle", "final"), full)
            },
            "budget": {
                "steps": steps,
                "pretrain_steps": pretrain_steps,
                "max_seconds": max_seconds,
            },
            "native_task": {
                name: _native_task(model, rows, sep)
                for name, rows in zip(("calibration", "oracle", "final"), full)
            },
        }
        artifact = None
        try:
            causal = decompile_coefficient(
                model,
                checkout,
                full[0],
                full[1],
                steps=steps,
                pretrain_steps=pretrain_steps,
                max_seconds=max_seconds,
                batch_size=8,
                min_fidelity=0.99,
                seed=0,
                max_mlp_rows=128,
                max_behavior_tests=1,
                max_feature_dim=256,
            )
            if fingerprint(model) != source:
                raise RuntimeError("coefficient search mutated source weights")
            artifact = causal.to_dict()
            case["causal_summary"] = summarize_stages(artifact)
            case["final_qk_replay"] = _replay_coefficient(
                model, artifact, checkout, full[2]
            )
            if fingerprint(model) != source:
                raise RuntimeError("coefficient replay mutated source weights")
        except Exception:  # noqa: BLE001 - persist this task, then continue the pilot
            case["task_error"] = traceback.format_exc()
        summary[task] = case
        write_json(
            out / f"{task}-summary.json",
            {key: value for key, value in case.items() if key != "artifact"},
        )
        if artifact is not None:
            write_json(out / f"{task}-stages.json", artifact)
        print(
            task,
            json.dumps(
                {
                    "accepted": None
                    if "causal_summary" not in case
                    else case["causal_summary"]["accepted"],
                    "stages": None
                    if "causal_summary" not in case
                    else [
                        {
                            "name": row["name"],
                            "accepted": row["accepted"],
                            "answer": row.get("evaluation_answer_agreement"),
                            "assumption": row.get("normalization_assumption_passed"),
                            "stop": row.get("stop"),
                        }
                        for row in case["causal_summary"]["stages"]
                    ],
                    "mlp": None
                    if artifact is None
                    else artifact["fidelity"].get("mlp_replacement"),
                    "task_error": "task_error" in case,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    write_json(
        out / "suite-index.json",
        {
            task: {
                "checkpoint": summary[task]["checkpoint"],
                "source_sha256": summary[task]["source_sha256"],
            }
            for task in summary
        },
    )
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?")
    p.add_argument("--checkout", required=True)
    p.add_argument("--component-only", action="store_true")
    p.add_argument("--calibration-steps", type=int, default=32)
    p.add_argument("--output", default="artifacts/trained-roundtrip")
    p.add_argument("--suite", action="store_true")
    p.add_argument("--coefficient", action="store_true")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--pretrain-steps", type=int, default=2)
    p.add_argument("--max-seconds", type=int, default=180)
    p.add_argument("--tasks", default="majority,copy,sort")
    args = p.parse_args()
    if args.coefficient:
        torch.set_num_threads(2)
        run_coefficient(
            args.checkout,
            args.output,
            [task for task in args.tasks.split(",") if task],
            args.steps,
            args.pretrain_steps,
            args.max_seconds,
        )
        return
    if args.suite:
        torch.set_num_threads(2)
        run_suite(
            args.checkout, args.output, [task for task in args.tasks.split(",") if task]
        )
        return
    if not args.model:
        raise ValueError("model path is required unless --suite is set")
    torch.set_num_threads(2)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("choose a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model)
    calibration = domain((1, 3, 5, 7))
    evaluation = domain((9,))
    driver = decompile if args.component_only else decompile_stages
    options = (
        {} if args.component_only else {"calibration_steps": args.calibration_steps}
    )
    causal = driver(
        model,
        args.checkout,
        calibration,
        max_trials=64,
        evaluation_rows=evaluation,
        **options,
    )
    write_json(out / "upstream.json", causal.to_dict())
    hypothesis = synthesize_binary(
        model, calibration, evaluation, causal_artifact=causal.to_dict()
    )
    write_json(out / "hypothesis.json", hypothesis.to_dict())
    if (
        hypothesis.nodes[-1].op != "count_threshold"
        or not hypothesis.fidelity["accepted"]
    ):
        raise ValueError("count rule was not recovered from this checkpoint")
    original = lower(hypothesis, 13)
    edited_artifact = hypothesis.to_dict()
    edited_artifact["nodes"][-1]["attributes"]["invert"] = True
    changed = lower(edited_artifact, 13)
    write_json(out / "before.rasp.json", original)
    write_json(out / "after.rasp.json", changed)
    # New-architecture path through the actual upstream compiler.
    tracr_before, _ = compile_program(original)
    tracr_after, _ = compile_program(changed)
    if not (out / "tracr-before").exists():
        tracr_before.save(out / "tracr-before")
    if not (out / "tracr-after").exists():
        tracr_after.save(out / "tracr-after")
    compiled_check = []
    for row in evaluation:
        seq = row["input_ids"]
        expected = int(sum(t == 1 for t in seq) < sum(t == 0 for t in seq))
        compiled_check.append(tracr_after.predict(seq)[-1] == expected)
    # Existing-architecture path: explicitly bounded constraint compilation.
    rule = {"op": "count_threshold", "threshold": 0.5, "invert": True}
    edited, report = cegis(model, rule, calibration, evaluation, heldout=domain((11,)))
    edited.save_pretrained(out / "patched")
    save_undo(model, edited, out / "patched")
    loaded = load_model(out / "patched")
    restored = undo(loaded, out / "patched")
    report["upstream_causal_fidelity"] = causal.fidelity
    report["symbolic_fidelity"] = hypothesis.fidelity
    report["tracr_new_model"] = {
        "checked": len(compiled_check),
        "accuracy": sum(compiled_check) / len(compiled_check),
        "architecture_preserved": False,
    }
    report["reload_changes"] = audit(edited, loaded)["changed_scalars"]
    report["undo_restored_original"] = fingerprint(restored) == fingerprint(model)
    write_json(out / "report.json", report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("coefficients", "direction", "upstream_causal_fidelity")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
