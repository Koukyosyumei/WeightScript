import argparse
import inspect
import json
from pathlib import Path

import torch

from .core import (
    audit,
    apply_raw_edits,
    compile_edits,
    extract,
    fingerprint,
    load_model,
    program_forward,
    write_json,
)
from .evaluate import benchmark, dataset, equivalence, score
from .checkpoint import save_undo, undo


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "workbench":
        del sys.argv[1]
        from .workbench import main as workbench

        return workbench()
    p = argparse.ArgumentParser(
        description="Inspect and edit trained GPT-2 circuits without gradients"
    )
    p.add_argument("--threads", type=int, default=2)
    sub = p.add_subparsers(dest="command", required=True)
    export = sub.add_parser(
        "extract", help="export editable computation IR and readable execution source"
    )
    export.add_argument("model")
    export.add_argument("output")
    edit = sub.add_parser(
        "compile", help="compile program edits into a reloadable HF checkpoint"
    )
    edit.add_argument("model")
    edit.add_argument("program")
    edit.add_argument("output")
    edit.add_argument(
        "--tokenizer", help="optional tokenizer to save alongside checkpoint"
    )
    raw = sub.add_parser(
        "compile-raw",
        help="direct tensor-edit baseline with the same audit and undo support",
    )
    raw.add_argument("model")
    raw.add_argument("program")
    raw.add_argument("output")
    raw.add_argument("--tokenizer")
    tensor = sub.add_parser(
        "tensor", help="read a bounded slice of a checkpoint tensor"
    )
    tensor.add_argument("model")
    tensor.add_argument("name")
    tensor.add_argument("--rows", default="0:8")
    tensor.add_argument("--cols", default="0:8")
    execute = sub.add_parser(
        "run", help="execute extracted/edited program without compiling weights"
    )
    execute.add_argument("model")
    execute.add_argument("program")
    execute.add_argument("tokens")
    restore = sub.add_parser(
        "undo", help="restore original tensors from a hash-checked undo package"
    )
    restore.add_argument("model")
    restore.add_argument("package")
    restore.add_argument("output")
    generate = sub.add_parser(
        "generate", help="greedy generation from text or token ids"
    )
    generate.add_argument("model")
    group = generate.add_mutually_exclusive_group(required=True)
    group.add_argument("--text")
    group.add_argument("--tokens")
    generate.add_argument("--tokenizer", help="tokenizer source (defaults to model)")
    generate.add_argument("--max-new-tokens", type=int, default=32)
    check = sub.add_parser(
        "verify", help="compare native, extracted and patched inference"
    )
    check.add_argument("model")
    check.add_argument("program")
    check.add_argument("data")
    check.add_argument("report")
    trace = sub.add_parser(
        "trace", help="attention patterns, activation summaries and weight statistics"
    )
    trace.add_argument("model")
    trace.add_argument("tokens", help="comma-separated token ids")
    trace.add_argument("output")
    ev = sub.add_parser("evaluate")
    ev.add_argument("model")
    ev.add_argument("data")
    ev.add_argument("report")
    bench = sub.add_parser("benchmark")
    bench.add_argument("model")
    bench.add_argument("candidates")
    bench.add_argument("dev")
    bench.add_argument("heldout")
    bench.add_argument("report")
    bench.add_argument("--budget", type=int, default=50)
    for parser in (check, ev, bench):
        parser.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    model = load_model(args.model)
    if args.command == "extract":
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "program.json", extract(model))
        write_json(output / "config.json", model.config.to_dict())
        write_json(
            output / "weights.json",
            {
                name: {
                    "shape": list(w.shape),
                    "norm": float(w.detach().norm()),
                    "min": float(w.detach().min()),
                    "max": float(w.detach().max()),
                }
                for name, w in model.named_parameters()
            },
        )
        # Source is for review and navigation; the supported editable surface is
        # program.json. Arbitrary Python edits are not claimed compilable.
        (output / "forward.py").write_text(
            "# Execution source for review. Edit program.json to compile supported operations.\n"
            + inspect.getsource(program_forward)
        )
        print(
            f"Extracted {len(extract(model)['blocks'])} layers, {sum(p.numel() for p in model.parameters()):,} parameters to {output}"
        )
    elif args.command in ("compile", "compile-raw"):
        program = json.loads(Path(args.program).read_text())
        if args.command == "compile":
            edited = compile_edits(model, program)
        elif "transaction" in program:
            from .raw import transaction

            edited = transaction(model, program["transaction"])
        else:
            edited = apply_raw_edits(model, program["raw"])
        output = Path(args.output)
        if output.exists() and any(output.iterdir()):
            raise ValueError(
                "output directory must be empty (source checkpoint is never overwritten)"
            )
        edited.save_pretrained(output, safe_serialization=True)
        save_undo(model, edited, output)
        if args.tokenizer:
            from transformers import AutoTokenizer

            AutoTokenizer.from_pretrained(
                args.tokenizer, trust_remote_code=False
            ).save_pretrained(output)
        write_json(output / "weightscript-program.json", program)
        write_json(output / "weightscript-audit.json", audit(model, edited))
        print(json.dumps(audit(model, edited), indent=2))
    elif args.command == "tensor":
        weights = model.state_dict()
        if args.name not in weights:
            raise ValueError("unknown tensor name; see extracted weights.json")
        w = weights[args.name]

        def bounds(text, size):
            start, stop = map(int, text.split(":"))
            if not 0 <= start < stop <= size or stop - start > 64:
                raise ValueError(
                    "slices must be in range and at most 64 elements per axis"
                )
            return slice(start, stop)

        if w.ndim == 2:
            selected = w[bounds(args.rows, w.shape[0]), bounds(args.cols, w.shape[1])]
        elif w.ndim == 1:
            selected = w[bounds(args.rows, w.shape[0])]
        else:
            raise ValueError("only vector and matrix inspection supported")
        print(
            json.dumps(
                {
                    "tensor": args.name,
                    "shape": list(w.shape),
                    "slice": selected.tolist(),
                },
                indent=2,
            )
        )
    elif args.command == "run":
        program = json.loads(Path(args.program).read_text())
        ids = torch.tensor([[int(t) for t in args.tokens.split(",")]])
        logits = program_forward(model, program, ids)
        print(
            json.dumps(
                {
                    "predicted_tokens": logits.argmax(-1).tolist(),
                    "last_logits": logits[0, -1].tolist(),
                }
            )
        )
    elif args.command == "undo":
        output = Path(args.output)
        if output.exists() and any(output.iterdir()):
            raise ValueError("output directory must be empty")
        restored = undo(model, args.package)
        restored.save_pretrained(output, safe_serialization=True)
        print(f"Restored original checkpoint: {fingerprint(restored)}")
    elif args.command == "generate":
        from transformers import AutoTokenizer

        tokenizer = None
        if args.text is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer or args.model, trust_remote_code=False
            )
            ids = tokenizer.encode(args.text, return_tensors="pt")
        else:
            ids = torch.tensor([[int(t) for t in args.tokens.split(",")]])
        if (
            not 0 < args.max_new_tokens <= model.config.n_positions - ids.shape[1]
            or not ids.numel()
        ):
            raise ValueError("generation must fit within model context length")
        with torch.inference_mode():
            for _ in range(args.max_new_tokens):
                next_token = (
                    model(ids, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
                )
                ids = torch.cat([ids, next_token], dim=1)
                if (
                    model.config.eos_token_id is not None
                    and int(next_token) == model.config.eos_token_id
                ):
                    break
        print(
            tokenizer.decode(ids[0], skip_special_tokens=True)
            if tokenizer
            else json.dumps(ids[0].tolist())
        )
    elif args.command == "verify":
        report = equivalence(
            model,
            json.loads(Path(args.program).read_text()),
            dataset(args.data),
            args.batch_size,
        )
        write_json(args.report, report)
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise SystemExit(1)
    elif args.command == "trace":
        ids = torch.tensor([[int(t) for t in args.tokens.split(",")]])
        program = extract(model)
        logits, traces = program_forward(model, program, ids, capture=True)
        rows = []
        for t in traces:
            scores = t["neurons"].abs().mean((0, 1))
            top = scores.topk(min(20, scores.numel()))
            a = t["attention"]
            i = t["layer"]
            binding = program["blocks"][i]["attention"]
            weights = model.state_dict()
            n_heads = binding["heads"]
            width = (
                (model.config.n_embd // n_heads)
                if model.config.model_type == "gpt2"
                else getattr(
                    model.config, "head_dim", model.config.hidden_size // n_heads
                )
            )
            heads = []
            for h in range(n_heads):
                sl = slice(h * width, (h + 1) * width)
                if model.config.model_type == "gpt2":
                    dim = model.config.n_embd
                    qkv = weights[binding["qkv"] + ".weight"]
                    q = qkv[:, sl]
                    k = qkv[:, dim + h * width : dim + (h + 1) * width]
                    v = qkv[:, 2 * dim + h * width : 2 * dim + (h + 1) * width]
                    o = weights[binding["write"] + ".weight"][sl]
                else:
                    kv_head = h // (n_heads // binding["kv_heads"])
                    kv_slice = slice(kv_head * width, (kv_head + 1) * width)
                    q = weights[binding["q"] + ".weight"][sl].T
                    k = weights[binding["k"] + ".weight"][kv_slice].T
                    v = weights[binding["v"] + ".weight"][kv_slice].T
                    o = weights[binding["write"] + ".weight"][:, sl].T
                heads.append(
                    {
                        "head": h,
                        "qk_singular_values": torch.linalg.svdvals(q @ k.T).tolist(),
                        "ov_singular_values": torch.linalg.svdvals(v @ o).tolist(),
                    }
                )
            rows.append(
                {
                    "layer": i,
                    "attention": a.tolist(),
                    "head_entropy": (
                        -(a.clamp_min(1e-30).log() * a).sum(-1).mean((0, 2))
                    ).tolist(),
                    "top_neurons": [
                        {"neuron": n, "mean_abs_activation": s}
                        for n, s in zip(top.indices.tolist(), top.values.tolist())
                    ],
                    "heads": heads,
                }
            )
        write_json(
            args.output,
            {
                "base_sha256": fingerprint(model),
                "tokens": ids.tolist(),
                "predicted_tokens": logits.argmax(-1).tolist(),
                "layers": rows,
            },
        )
        print(f"Wrote circuit trace to {args.output}")
    elif args.command == "evaluate":
        report = score(model, dataset(args.data), args.batch_size)
        write_json(args.report, report)
        print(json.dumps(report, indent=2))
    elif args.command == "benchmark":
        report = benchmark(
            model,
            json.loads(Path(args.candidates).read_text()),
            dataset(args.dev),
            dataset(args.heldout),
            args.batch_size,
            args.budget,
        )
        write_json(args.report, report)
        for method in report["methods"]:
            print(
                method["method"],
                "heldout:",
                method["heldout"],
                "changed:",
                method["audit"]["changed_scalars"],
            )


if __name__ == "__main__":
    main()
