"""A real local instruction-model policy for the agent protocol (CPU).

Its proposed edits can fail. This is a runnable policy backend, not a claim that
small instruction models match the frontier agents in the motivating research.
"""

import argparse
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
p.add_argument("--max-new-tokens", type=int, default=256)
p.add_argument("--threads", type=int, default=2)
a = p.parse_args()
request = json.load(sys.stdin)
torch.set_num_threads(a.threads)
context = request["context"]
# Exclude verbose graph/config copies; numerical tensors remain accessible through tools.
compact = {
    k: context[k]
    for k in (
        "goal",
        "mode",
        "seed",
        "checkpoint_sha256",
        "weights",
        "baseline_dev",
        "actions",
        "proposal_interfaces",
        "action_examples",
    )
}
compact["transaction_schema"] = context.get("transaction_schema")
compact["development_examples"] = context["development_examples"][:16]
if context["mode"] != "direct":
    compact["rule_schema"] = context.get("rule_schema")
    compact["program"] = context.get("program")
compact.update(
    step=request["step"],
    history=request["history"],
    remaining_forwards=request["remaining_forwards"],
)
tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
model = (
    AutoModelForCausalLM.from_pretrained(
        a.model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        trust_remote_code=False,
    )
    .cpu()
    .eval()
)
messages = [
    {
        "role": "system",
        "content": 'You edit neural network weights without gradients. Return exactly one JSON action object and no explanation. Supported actions: inspect, trace, evaluate, finish. An evaluate action has a proposal object using only the allowed interfaces. A finish action is {"action":"finish"}. Choose a targeted edit to achieve the goal; inspect if necessary. Never invent tensor names.',
    },
    {"role": "user", "content": json.dumps(compact)},
]
ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
)
start = time.monotonic()
with torch.inference_mode():
    out = model.generate(
        ids,
        max_new_tokens=a.max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
text = tokenizer.decode(out[0, ids.shape[-1] :], skip_special_tokens=True)
try:
    begin = text.index("{")
    answer, _ = json.JSONDecoder().raw_decode(text[begin:])
except (ValueError, json.JSONDecodeError):
    # Preserve failure rather than silently inventing a better proposal.
    answer = {"action": "invalid", "raw_response": text}
answer["usage"] = {
    "model": a.model,
    "revision": getattr(model.config, "_commit_hash", None),
    "input_tokens": ids.numel(),
    "output_tokens": out.numel() - ids.numel(),
    "generation_seconds": time.monotonic() - start,
    "external_model_forward_calls": out.shape[-1] - ids.shape[-1],
}
print(json.dumps(answer))
