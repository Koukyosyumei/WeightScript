"""Unified research CLI for semantic, native and direct-weight workflows."""

import argparse
import json
import shlex
from pathlib import Path

import torch

from .core import load_model, write_json
from .evaluate import dataset


def main():
    parser = argparse.ArgumentParser(description="WeightScript research workbench")
    parser.add_argument("--threads", type=int, default=2)
    sub = parser.add_subparsers(dest="command", required=True)
    explanation = sub.add_parser("explain")
    explanation.add_argument("model")
    explanation.add_argument("edited")
    explanation.add_argument("output")
    gen = sub.add_parser("evaluate-generation")
    gen.add_argument("model")
    gen.add_argument("data")
    gen.add_argument("output")
    gen.add_argument("--max-new-tokens", type=int, default=32)
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--seed", type=int, default=0)
    rewrite = sub.add_parser(
        "rewrite",
        help="counterexample-guided bounded rule compilation into trained weights",
    )
    rewrite.add_argument("model")
    rewrite.add_argument("rule")
    rewrite.add_argument("synthesis")
    rewrite.add_argument("oracle")
    rewrite.add_argument("output")
    rewrite.add_argument("--heldout")
    rewrite.add_argument("--max-rounds", type=int, default=4)
    replay = sub.add_parser("run-decompiled")
    replay.add_argument("model")
    replay.add_argument("artifact")
    replay.add_argument("tokens", help="JSON token list")
    replay.add_argument("--checkout", required=True)
    usage = sub.add_parser("codex-usage")
    usage.add_argument("workspace")
    usage.add_argument("output")
    tui = sub.add_parser(
        "codex-start", help="start a separate authenticated Codex TUI in tmux"
    )
    tui.add_argument("workspace")
    tui.add_argument("--session", default="weightscript-editor")
    tui.add_argument("--model")
    graph = sub.add_parser("graph")
    graph.add_argument("model")
    graph.add_argument("output")
    dec = sub.add_parser(
        "decompile", help="real upstream component pruning and source generation"
    )
    dec.add_argument("model")
    dec.add_argument("calibration")
    dec.add_argument("output")
    dec.add_argument("--checkout", required=True)
    dec.add_argument("--evaluation")
    dec.add_argument("--max-trials", type=int, default=64)
    dec.add_argument("--min-fidelity", type=float, default=0.99)
    dec.add_argument("--linear-ln", action="store_true")
    synth = sub.add_parser(
        "synthesize", help="domain-limited behavioral primitive hypotheses"
    )
    synth.add_argument("model")
    synth.add_argument("calibration")
    synth.add_argument("output")
    synth.add_argument("--evaluation")
    synth.add_argument("--causal-artifact")
    synth.add_argument("--max-seq-len", type=int, default=17)
    rp = sub.add_parser("rasp")
    rs = rp.add_subparsers(dest="operation", required=True)
    for op in ("compile", "patch", "verify", "run", "recover"):
        p = rs.add_parser(op)
        if op != "compile":
            p.add_argument("checkpoint")
        if op in ("compile", "patch", "verify"):
            p.add_argument("program")
        if op in ("compile", "patch", "recover"):
            p.add_argument("output")
        if op == "recover":
            p.add_argument("--library", required=True)
        if op == "run":
            p.add_argument("tokens", help="JSON list")
        if op == "verify":
            p.add_argument("--max-len", type=int, default=4)
    agent = sub.add_parser("session", help="budgeted external-agent loop")
    agent.add_argument("model")
    agent.add_argument("dev")
    agent.add_argument("output")
    agent.add_argument(
        "--policy",
        required=True,
        help="trusted policy command, JSON request on stdin and response on stdout",
    )
    agent.add_argument("--goal", required=True)
    agent.add_argument(
        "--mode", choices=["direct", "program", "hybrid"], default="hybrid"
    )
    agent.add_argument("--heldout")
    agent.add_argument("--collateral")
    agent.add_argument("--seed", type=int, default=0)
    agent.add_argument("--max-forwards", type=int, default=50)
    agent.add_argument("--max-steps", type=int, default=30)
    agent.add_argument("--policy-timeout", type=int, default=60)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.command == "rasp":
        from .backends.tracr import Compiled, compile_program, patch, verify, recover

        if args.operation == "compile":
            spec = json.loads(Path(args.program).read_text())
            compiled, _ = compile_program(spec)
            compiled.save(args.output)
            print(compiled.fingerprint())
        elif args.operation == "patch":
            model = Compiled.load(args.checkpoint)
            spec = json.loads(Path(args.program).read_text())
            edited, report = patch(model, spec)
            edited.save(args.output)
            write_json(Path(args.output) / "audit.json", report)
            print(json.dumps(report, indent=2))
        elif args.operation == "verify":
            report = verify(
                Compiled.load(args.checkpoint),
                json.loads(Path(args.program).read_text()),
                args.max_len,
            )
            print(json.dumps(report, indent=2))
            if not report["passed"]:
                raise SystemExit(1)
        elif args.operation == "run":
            print(
                json.dumps(
                    Compiled.load(args.checkpoint).predict(json.loads(args.tokens))
                )
            )
        elif args.operation == "recover":
            result = recover(
                Compiled.load(args.checkpoint),
                json.loads(Path(args.library).read_text()),
            )
            write_json(args.output, result)
        return
    if args.command == "codex-usage":
        from .tmux_policy import read_usage

        result = read_usage(args.workspace)
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return
    if args.command == "codex-start":
        from .tmux_policy import start

        print(json.dumps(start(args.workspace, args.session, args.model), indent=2))
        return
    model = load_model(args.model)
    if args.command == "run-decompiled":
        from .backends.upstream import execute_artifact

        logits = execute_artifact(
            model,
            json.loads(Path(args.artifact).read_text()),
            args.checkout,
            torch.tensor([json.loads(args.tokens)], dtype=torch.long),
        )
        print(
            json.dumps(
                {
                    "predicted_tokens": logits.argmax(-1).tolist(),
                    "last_logits": logits[0, -1].tolist(),
                }
            )
        )
    elif args.command == "explain":
        from .explain import explain

        write_json(args.output, explain(model, load_model(args.edited)).to_dict())
    elif args.command == "evaluate-generation":
        from .generation import evaluate_generation

        report = evaluate_generation(
            model, dataset(args.data), args.max_new_tokens, args.temperature, args.seed
        )
        write_json(args.output, report)
        print(json.dumps(report, indent=2))
    elif args.command == "rewrite":
        from .synthesis import cegis
        from .checkpoint import save_undo

        output = Path(args.output)
        if output.exists() and any(output.iterdir()):
            raise ValueError("output must be empty")
        edited, report = cegis(
            model,
            json.loads(Path(args.rule).read_text()),
            dataset(args.synthesis),
            dataset(args.oracle),
            dataset(args.heldout) if args.heldout else None,
            args.max_rounds,
        )
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "report.json", report)
        if not report["bounded_oracle_passed"]:
            raise SystemExit("bounded compilation failed; checkpoint not exported")
        edited.save_pretrained(output / "checkpoint")
        save_undo(model, edited, output / "checkpoint")
        print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
    elif args.command == "graph":
        from .ir import graph_from_model

        write_json(args.output, graph_from_model(model).to_dict())
    elif args.command == "decompile":
        from .backends.upstream import decompile

        result = decompile(
            model,
            args.checkout,
            dataset(args.calibration),
            args.max_trials,
            args.min_fidelity,
            linear_ln=args.linear_ln,
            evaluation_rows=dataset(args.evaluation) if args.evaluation else None,
        )
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "artifact.json", result.to_dict())
        (output / "program.txt").write_text(
            result.nodes[-1].attributes["source"] + "\n"
        )
        print(json.dumps(result.fidelity, indent=2))
    elif args.command == "synthesize":
        from .symbolic import synthesize_binary, lower

        causal = (
            json.loads(Path(args.causal_artifact).read_text())
            if args.causal_artifact
            else None
        )
        result = synthesize_binary(
            model,
            dataset(args.calibration),
            dataset(args.evaluation) if args.evaluation else None,
            causal_artifact=causal,
        )
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "artifact.json", result.to_dict())
        if result.fidelity["accepted"]:
            try:
                write_json(
                    output / "program.rasp.json", lower(result, args.max_seq_len)
                )
            except ValueError as error:
                print(str(error))
        print(json.dumps(result.fidelity, indent=2))
    elif args.command == "session":
        from .agent import run_session

        result = run_session(
            model,
            shlex.split(args.policy),
            args.goal,
            dataset(args.dev),
            dataset(args.heldout) if args.heldout else None,
            mode=args.mode,
            max_forwards=args.max_forwards,
            max_steps=args.max_steps,
            timeout=args.policy_timeout,
            seed=args.seed,
            output=args.output,
            collateral=dataset(args.collateral) if args.collateral else None,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
