"""Unified research CLI for semantic, native and direct-weight workflows."""

import argparse
import json
import shlex
from pathlib import Path

from ._cli_support import run_cli, positive_int, add_version


def build_parser(subparsers=None):
    parser = argparse.ArgumentParser(description="WeightScript research workbench")
    add_version(parser)
    parser.add_argument("--threads", type=positive_int, default=2)
    sub = (
        subparsers
        if subparsers is not None
        else parser.add_subparsers(dest="command", required=True)
    )
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
    replay = sub.add_parser(
        "run-decompiled",
        help="replay a saved component, full-path, or QK artifact",
    )
    replay.add_argument("model")
    replay.add_argument("artifact")
    replay.add_argument("tokens", help="JSON token list")
    replay.add_argument("--checkout", required=True)
    replay.add_argument(
        "--stage",
        choices=("component", "full_path", "qk"),
        default="component",
        help="component reads a stage-1 artifact, or the nested component stage of a three-stage artifact",
    )
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
        "decompile",
        help="run a pinned upstream engine and write its artifact",
        description=(
            "The component engine is the default. --engine three-stage also runs "
            "full-path and QK pruning and weight-level QK matching. The accepted "
            "flag records final-token argmax agreement on the supplied rows. "
            "Full-span replay and logit error are separate fields. This is not "
            "the paper coefficient search or MLP primitive replacement."
        ),
    )
    dec.add_argument("model")
    dec.add_argument("calibration")
    dec.add_argument("output")
    dec.add_argument("--checkout", required=True)
    dec.add_argument("--evaluation")
    dec.add_argument("--max-trials", type=int, default=64)
    dec.add_argument("--min-fidelity", type=float, default=0.99)
    dec.add_argument("--linear-ln", action="store_true")
    dec.add_argument(
        "--engine", choices=("component", "three-stage"), default="component"
    )
    dec.add_argument(
        "--calibration-steps",
        type=int,
        default=None,
        help="three-stage auxiliary-vector steps; default 32 for that engine",
    )
    synth = sub.add_parser(
        "synthesize",
        help="match an explicit finite rule library on calibration rows only",
    )
    synth.add_argument("model")
    synth.add_argument("calibration")
    synth.add_argument("output")
    synth.add_argument("--evaluation")
    synth.add_argument(
        "--final", help="sequence synthesis report rows; not used for selection"
    )
    synth.add_argument("--candidates", help="JSON list of explicit rules")
    synth.add_argument(
        "--sequence",
        action="store_true",
        help="score span_next programs at every supervised position",
    )
    synth.add_argument("--causal-artifact")
    synth.add_argument("--min-fidelity", type=float, default=1.0)
    synth.add_argument("--max-seq-len", type=int, default=17)
    checked = sub.add_parser("rule-check", help="validate an explicit semantic rule")
    checked.add_argument("rule")
    predicted = sub.add_parser(
        "rule-predict", help="execute a validated rule on one token list"
    )
    predicted.add_argument("rule")
    predicted.add_argument("tokens", help="JSON list of token ids")
    pipe = sub.add_parser(
        "pipeline",
        help="three-stage extraction, explicit rule match, then a bounded edit",
        description=(
            "Final rows are scored only after refinement stops. A rejected stage "
            "or hypothesis writes the report and does not export a checkpoint."
        ),
    )
    pipe.add_argument("model")
    pipe.add_argument("--checkout", required=True)
    pipe.add_argument("--candidates", required=True)
    pipe.add_argument("--rule", required=True)
    pipe.add_argument("--synthesis", required=True)
    pipe.add_argument("--oracle", required=True)
    pipe.add_argument("--final", required=True)
    pipe.add_argument("output")
    pipe.add_argument("--max-trials", type=int, default=64)
    pipe.add_argument("--max-rounds", type=int, default=4)
    pipe.add_argument("--min-fidelity", type=float, default=1.0)
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
    agent.add_argument(
        "--objective",
        choices=["next_token", "sequence"],
        default="next_token",
        help="next_token scores one target; sequence scores every target token",
    )
    agent.add_argument("--seed", type=int, default=0)
    agent.add_argument("--max-forwards", type=int, default=50)
    agent.add_argument("--max-steps", type=int, default=30)
    agent.add_argument("--policy-timeout", type=int, default=60)
    agent.add_argument(
        "--resume",
        action="store_true",
        help="recover a durable session from its output directory",
    )
    started = sub.add_parser(
        "agent-start",
        help="start a detached Grok or Codex editing TUI; does not fabricate a reply",
    )
    started.add_argument("workspace")
    started.add_argument("--provider", choices=["grok", "codex"], default="grok")
    started.add_argument("--session", default="weightscript-editor")
    started.add_argument("--model")
    agent_usage = sub.add_parser(
        "agent-usage",
        help="collect provider-reported usage without inventing token totals",
    )
    agent_usage.add_argument("output")
    agent_usage.add_argument("--provider", choices=["grok", "codex"], required=True)
    agent_usage.add_argument("--session-id", help="Grok session UUID")
    agent_usage.add_argument("--workspace", help="Codex workspace directory")
    return parser


def _rule_command(args):
    from .backends.rules import rule_predict, validate_rule

    rule = json.loads(Path(args.rule).read_text())
    validate_rule(rule)
    if args.command == "rule-check":
        print(json.dumps({"valid": True, "op": rule["op"]}))
        return
    tokens = json.loads(args.tokens)
    if not isinstance(tokens, list):
        raise ValueError("tokens must be a JSON list")
    print(json.dumps({"token": rule_predict(rule, tokens)}))


def execute(args):
    if args.command in ("rule-check", "rule-predict"):
        _rule_command(args)
        return
    import torch
    from .core import load_model, write_json
    from .evaluate import dataset

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
    if args.command == "agent-start":
        if args.provider == "grok":
            from .agent_bridge import start_grok

            result = start_grok(args.workspace, args.session, args.model)
        else:
            from .tmux_policy import start

            result = start(args.workspace, args.session, args.model)
        print(json.dumps(result, indent=2))
        return
    if args.command == "agent-usage":
        if args.provider == "grok":
            if not args.session_id:
                raise ValueError("grok usage requires --session-id")
            from .agent_bridge import read_grok_usage

            result = read_grok_usage(args.session_id)
        else:
            if not args.workspace:
                raise ValueError("codex usage requires --workspace")
            from .tmux_policy import read_usage

            result = read_usage(args.workspace)
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return
    model = load_model(args.model)
    if args.command == "run-decompiled":
        artifact = json.loads(Path(args.artifact).read_text())
        ids = torch.tensor([json.loads(args.tokens)], dtype=torch.long)
        if args.stage == "component":
            from .backends.upstream import execute_artifact

            if artifact.get("backend") == "decompiling_transformers.three_stage":
                artifact = next(
                    item["artifact"]
                    for item in artifact["nodes"][-1]["attributes"]["stages"]
                    if item["name"] == "component"
                )
            logits = execute_artifact(model, artifact, args.checkout, ids)
        else:
            from .backends.upstream_stages import execute_stages

            logits = execute_stages(model, artifact, args.checkout, ids, args.stage)
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
        evaluation = dataset(args.evaluation) if args.evaluation else None
        calibration = dataset(args.calibration)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        if args.engine == "three-stage":
            from .backends.pipeline import summarize_stages
            from .backends.upstream_stages import decompile_stages

            result = decompile_stages(
                model,
                args.checkout,
                calibration,
                evaluation_rows=evaluation,
                max_trials=args.max_trials,
                min_fidelity=args.min_fidelity,
                calibration_steps=32
                if args.calibration_steps is None
                else args.calibration_steps,
            )
            summary = summarize_stages(result)
            write_json(output / "artifact.json", result.to_dict())
            write_json(output / "summary.json", summary)
            print(json.dumps(summary, indent=2))
        else:
            if args.calibration_steps is not None:
                raise ValueError("--calibration-steps applies to --engine three-stage")
            from .backends.upstream import decompile

            result = decompile(
                model,
                args.checkout,
                calibration,
                args.max_trials,
                args.min_fidelity,
                linear_ln=args.linear_ln,
                evaluation_rows=evaluation,
            )
            write_json(output / "artifact.json", result.to_dict())
            (output / "program.txt").write_text(
                result.nodes[-1].attributes["source"] + "\n"
            )
            print(json.dumps(result.fidelity, indent=2))
    elif args.command == "pipeline":
        from .backends.pipeline import decompile_and_edit
        from .checkpoint import save_undo

        output = Path(args.output)
        if output.exists() and any(output.iterdir()):
            raise ValueError("output must be empty")
        output.mkdir(parents=True, exist_ok=True)
        edited, report = decompile_and_edit(
            model,
            args.checkout,
            json.loads(Path(args.candidates).read_text()),
            json.loads(Path(args.rule).read_text()),
            dataset(args.synthesis),
            dataset(args.oracle),
            dataset(args.final),
            max_trials=args.max_trials,
            max_rounds=args.max_rounds,
            min_fidelity=args.min_fidelity,
        )
        write_json(output / "report.json", report)
        if edited is None:
            raise ValueError(
                "semantic pipeline wrote report.json with status " + report["status"]
            )
        edited.save_pretrained(output / "checkpoint")
        save_undo(model, edited, output / "checkpoint")
        print(report["status"])
    elif args.command == "synthesize":
        causal = (
            json.loads(Path(args.causal_artifact).read_text())
            if args.causal_artifact
            else None
        )
        if args.candidates:
            from .symbolic import synthesize_rules, synthesize_sequence_rules

            candidates = json.loads(Path(args.candidates).read_text())
            if args.sequence:
                result = synthesize_sequence_rules(
                    model,
                    candidates,
                    dataset(args.calibration),
                    dataset(args.evaluation) if args.evaluation else None,
                    dataset(args.final) if args.final else None,
                    min_fidelity=args.min_fidelity,
                    causal_artifact=causal,
                )
            else:
                if args.final:
                    raise ValueError("--final applies to --sequence synthesis")
                result = synthesize_rules(
                    model,
                    candidates,
                    dataset(args.calibration),
                    dataset(args.evaluation) if args.evaluation else None,
                    min_fidelity=args.min_fidelity,
                    causal_artifact=causal,
                )
        else:
            if args.sequence or args.final:
                raise ValueError("--sequence requires --candidates")
            from .symbolic import synthesize_binary

            result = synthesize_binary(
                model,
                dataset(args.calibration),
                dataset(args.evaluation) if args.evaluation else None,
                causal_artifact=causal,
            )
        from .symbolic import lower

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
            resume=args.resume,
            objective=args.objective,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


def main(argv=None):
    """Compatibility entry point for the original research command surface."""
    args = build_parser().parse_args(argv)
    return run_cli(lambda: execute(args))


if __name__ == "__main__":
    raise SystemExit(main())
