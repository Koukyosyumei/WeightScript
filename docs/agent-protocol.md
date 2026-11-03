# Agent protocol

An external command reads one request JSON object from stdin, writes one response JSON object to stdout, and exits. It is invoked separately each turn. Its logs belong on stderr. The controller caps response/log sizes at 1 MiB, applies a per-command timeout, and kills the process group on timeout. This is a trusted-process interface, not filesystem/network isolation.

Requests contain `context`, `step`, `history`, `remaining_forwards`, `remaining_steps`, `run_id` and `request_nonce`. Context provides the goal, mode, checkpoint hash, parameter names/shapes, development examples, baseline metrics, allowed proposal interfaces and action examples. Program and graph context are absent in direct mode. Held-out and collateral rows are never included in requests. A response that sets `request_nonce` or `run_id` must echo the request. Omitted nonce fields stay compatible with older policies. The same nonce is reused if the controller restarts while that request is still pending.

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

The session objective is `next_token` or `sequence`. `next_token` scores the single `target` at the last prompt position. `sequence` scores every id in `targets` for development selection, held-out, and collateral, and rejects a copy or sort row whose labels stop before the final answer token. A first-token match is not a full-span success. The configuration digest includes the objective, so a resume that changes it fails. Held-out and collateral rows stay out of the policy request.

For trustworthy comparisons, use a policy process that does not access benchmark files outside the protocol, separate development/counterexample/final domains, match model/context/tool/time budgets, record the protocol before executing both arms, and report failures and repeated seeds. The protocol record is not public preregistration or proof against a malicious policy.

## External TUI policy

The editing target can be a small local model while the editing agent is a separate Grok or Codex CLI session. Each CLI uses its own existing login. WeightScript does not collect or invent credentials.

```bash
weightscript agent-start ./my-editing-agent --provider grok --session my-editing-agent
weightscript session MODEL dev.jsonl artifacts/agent-session \
  --policy 'python -m weightscript.tmux_policy --provider grok --workspace ./my-editing-agent --session my-editing-agent' \
  --goal 'Desired model improvement' --mode hybrid --heldout heldout.jsonl \
  --policy-timeout 260
weightscript agent-usage artifacts/agent-usage.json --provider grok --session-id SESSION_UUID
```

`agent-start --provider grok` reads `grok --help` and launches only flags that help text advertises, including a workspace-local leader socket so the editor does not attach to another Grok session. It writes `provider.json` with the CLI version and session id. It does not scrape a reply or mark the process authenticated. `codex-start` remains the Codex-specific launcher. A fresh workspace is required for each independent arm.

An identical request file is retried without sending a second prompt once delivery was recorded. If the response file is already present, it is returned. A different request for the same step is rejected.

The bridge writes `request-NNNN.json`, submits one prompt to the existing TUI, and waits for that provider to write `response-NNNN.json` atomically. No model response is fabricated or scraped from terminal text. The agent can inspect the supplied files, while model access and evaluation run through controller actions. Separate AGENTS.md instructions prohibit benchmark access and unmetered inference; this is a trusted-agent protocol, not technical confinement.

`tmux attach -t my-editing-agent` shows the live editing agent. The tmux session is created detached, with `destroy-unattached` off, so closing the client does not by itself stop it. `codex-usage` reports best-effort cumulative local rollout usage for a Codex workspace. `agent-usage --provider grok` stores the verbatim `grok usage` output and does not fill in totals the CLI did not print.
