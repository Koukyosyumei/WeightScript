# Agent protocol

An external command reads one request JSON object from stdin, writes one response JSON object to stdout, and exits. It is invoked separately each turn. Its logs belong on stderr. The controller caps response/log sizes at 1 MiB, applies a per-command timeout, and kills the process group on timeout. This is a trusted-process interface, not filesystem/network isolation.

Requests contain `context`, `step`, `history`, `remaining_forwards`, and `remaining_steps`. Context provides the goal, mode, checkpoint hash, parameter names/shapes, development examples, baseline metrics, allowed proposal interfaces and action examples. Program and graph context are absent in direct mode. Held-out and collateral rows are never included in requests.

Responses:

```json
{"action":"inspect","tensor":"transformer.h.0.attn.c_proj.weight","start":0,"stop":2,"col_start":0,"col_stop":4}
```

Inspection always reads source checkpoint weights. Vector/matrix slices are bounded to 32 entries per axis.

```json
{"action":"trace","input_ids":[2,0,1,0,3]}
{"action":"generate","input_ids":[2,0,1,0,3],"max_new_tokens":8}
```

Trace and generation use the latest successfully evaluated candidate. Trace includes attention and top neuron activations. Each trace costs one inference reservation; generation reserves the requested tokens and refunds unexecuted steps after EOS.

```json
{"action":"evaluate","proposal":{"edits":[{"op":"head_scale","layer":0,"head":0,"factor":0.5}]}}
{"action":"evaluate","proposal":{"transaction":[{"op":"scale","tensor":"transformer.h.0.attn.c_proj.weight","select":[[0,1],null],"value":0.5}]}}
{"action":"evaluate","proposal":{"rule":{"op":"count_threshold","threshold":0.5,"invert":true}}}
{"action":"finish"}
```

Every proposal is relative to the original model. Direct mode accepts legacy `raw` operations and generic `transaction` operations. Program mode accepts adapter `edits` and bounded `rule` compilation. Hybrid mode composes rule, adapter edits, raw edits and transactions in that order. The built-in rule compiler currently supports binary count thresholds in GPT-2 and uses the development inputs as its explicit synthesis domain; its native-model probes consume inference allowance.

An optional `usage` response records external-model tokens, calls and time. It is policy-reported rather than independently enforced. The local policy includes the model revision and token counts. The framework's conservative reservations can charge invalid operations which fail before inference; these are not asserted to be exact hardware call counts.

The best development candidate is retained by accuracy, then NLL; a bad final proposal cannot replace a better candidate. Invalid actions are journaled and returned as feedback. On termination, the controller evaluates the selected candidate on held-out and collateral domains, saves an ordinary checkpoint, an audit and undo tensors. Policy failures do not silently synthesize replacement edits.

For trustworthy comparisons, use a policy process that does not access benchmark files outside the protocol, separate development/counterexample/final domains, match model/context/tool/time budgets, record the protocol before executing both arms, and report failures and repeated seeds. The protocol record is not public preregistration or proof against a malicious policy.

## Codex TUI policy (preferred editing agent)

The editing target can be a small local model while the editing agent is an authenticated Codex CLI session. This uses the CLI's existing authentication; it does not require an API key in the experiment process.

```bash
weightscript codex-start /dev/shm/my-weightscript-agent --session my-weightscript-agent
weightscript session MODEL dev.jsonl artifacts/codex-session \
  --policy 'python -m weightscript.tmux_policy --workspace /dev/shm/my-weightscript-agent --session my-weightscript-agent' \
  --goal 'Desired model improvement' --mode hybrid --heldout heldout.jsonl \
  --policy-timeout 260
weightscript codex-usage /dev/shm/my-weightscript-agent artifacts/codex-usage.json
```

`codex-start` creates a fresh workspace, task instructions and a standalone TUI (`--no-daemon`) in tmux. It inherits the configured Codex model unless `--model` is explicitly supplied. Standalone launch avoids attaching a comparison arm to another active daemon thread. Verify the TUI's working directory/session before starting an experiment. A fresh workspace is required for each independent arm.

The bridge writes immutable `request-NNNN.json`, submits a prompt to the existing TUI, and waits for Codex to write `response-NNNN.json` atomically. No model response is fabricated or scraped from terminal text. Codex can inspect the supplied files and compute with observed tensors, while model access and evaluation run through controller actions. Separate AGENTS.md instructions prohibit benchmark access and unmetered inference; this is a trusted-agent protocol, not technical confinement.

`tmux attach -t my-weightscript-agent` shows the live editing agent. The TUI process and its conversation survive controller steps. `codex-usage` reports best-effort cumulative local rollout usage for that specific working directory; it includes any debugging or later turns in that thread and is not a per-step invoice.
