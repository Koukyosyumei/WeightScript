"""Independent dense decoder adapters for Llama and Qwen2 (default RoPE only)."""

import copy
import math

import torch
import torch.nn.functional as F
from transformers.activations import ACT2FN

SUPPORTED = {"llama", "qwen2"}


def validate(model):
    c = model.config
    if c.model_type not in SUPPORTED:
        raise ValueError("unsupported decoder architecture")
    if getattr(c, "rope_scaling", None) not in (None, {}):
        raise ValueError("scaled/dynamic RoPE needs a separate adapter")
    if getattr(c, "use_sliding_window", False) or getattr(c, "sliding_window", None):
        # Llama config normally has no window; Qwen2 may store a window while disabled.
        if c.model_type != "qwen2" or getattr(c, "use_sliding_window", False):
            raise ValueError("sliding-window attention is unsupported")
    if getattr(c, "partial_rotary_factor", 1.0) != 1.0:
        raise ValueError("partial rotary dimensions unsupported")
    if model.training or any(
        p.dtype != torch.float32 or p.device.type != "cpu" for p in model.parameters()
    ):
        raise ValueError("adapter requires eval-mode float32 CPU model")
    if type(model).__name__ not in {"LlamaForCausalLM", "Qwen2ForCausalLM"}:
        raise ValueError("custom decoder subclasses are unsupported")


def extract(model):
    from .core import fingerprint

    validate(model)
    c = model.config
    return {
        "format": f"weightscript-{c.model_type}-v1",
        "base_sha256": fingerprint(model),
        "semantics": "float32 causal inference; RMSNorm; default RoPE; GQA; no KV cache",
        "blocks": [
            {
                "layer": i,
                "attention": {
                    "op": "rope_gqa_attention",
                    "heads": c.num_attention_heads,
                    "kv_heads": c.num_key_value_heads,
                    "q": f"model.layers.{i}.self_attn.q_proj",
                    "k": f"model.layers.{i}.self_attn.k_proj",
                    "v": f"model.layers.{i}.self_attn.v_proj",
                    "write": f"model.layers.{i}.self_attn.o_proj",
                },
                "mlp": {
                    "op": c.hidden_act,
                    "gate": f"model.layers.{i}.mlp.gate_proj",
                    "up": f"model.layers.{i}.mlp.up_proj",
                    "write": f"model.layers.{i}.mlp.down_proj",
                },
                "norms": [
                    f"model.layers.{i}.input_layernorm",
                    f"model.layers.{i}.post_attention_layernorm",
                ],
            }
            for i in range(c.num_hidden_layers)
        ],
        "edits": [],
    }


def validate_program(model, program):
    base = extract(model)
    if set(program) != set(base) or any(
        program[k] != base[k] for k in base if k != "edits"
    ):
        raise ValueError("program metadata does not match decoder checkpoint")
    if not isinstance(program["edits"], list):
        raise ValueError("edits must be a list")
    for e in program["edits"]:
        op = e.get("op")
        fields = {"op", "layer"}
        if op in {"head_scale", "attention_scale"}:
            fields |= {"head", "factor"}
        elif op == "neuron_scale":
            fields |= {"neurons", "factor"}
        elif op == "remove_direction":
            fields |= {"neurons", "direction", "strength"}
        else:
            raise ValueError("unknown decoder edit")
        if (
            set(e) != fields
            or type(e["layer"]) is not int
            or not 0 <= e["layer"] < model.config.num_hidden_layers
        ):
            raise ValueError("invalid decoder edit fields or layer")
        if "head" in e and (
            type(e["head"]) is not int
            or not 0 <= e["head"] < model.config.num_attention_heads
        ):
            raise ValueError("head outside range")
        if "neurons" in e and (
            not isinstance(e["neurons"], list)
            or not e["neurons"]
            or len(set(e["neurons"])) != len(e["neurons"])
            or any(
                type(n) is not int or not 0 <= n < model.config.intermediate_size
                for n in e["neurons"]
            )
        ):
            raise ValueError("invalid neuron indices")
        f = e.get("factor", e.get("strength"))
        if type(f) not in (int, float) or not math.isfinite(f) or abs(f) > 1e4:
            raise ValueError("invalid factor")
        if op == "remove_direction":
            d = torch.tensor(e["direction"], dtype=torch.float32)
            if (
                d.shape != (model.config.hidden_size,)
                or not torch.isfinite(d).all()
                or not torch.isfinite(d.norm())
                or d.norm() == 0
            ):
                raise ValueError("invalid residual direction")


@torch.inference_mode()
def forward(model, program, ids, attention_mask=None, capture=False, validated=False):
    if not validated:
        validate_program(model, program)
    c = model.config
    if (
        ids.ndim != 2
        or not ids.numel()
        or ids.shape[1] > c.max_position_embeddings
        or ids.min() < 0
        or ids.max() >= c.vocab_size
    ):
        raise ValueError("invalid token domain")
    w = model.state_dict()
    x = F.embedding(ids, w["model.embed_tokens.weight"])
    length = ids.shape[1]
    head_dim = getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)
    positions = torch.arange(length, dtype=torch.float32)
    inv = 1.0 / (
        c.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    frequencies = positions[:, None] * inv[None, :]
    rotary = torch.cat((frequencies, frequencies), -1)
    cos = rotary.cos()[None, None]
    sin = rotary.sin()[None, None]

    def rotate(z):
        first, second = z.chunk(2, -1)
        return torch.cat((-second, first), -1)

    def norm(z, prefix):
        return (
            z
            * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + c.rms_norm_eps)
            * w[prefix + ".weight"]
        )

    def linear(z, prefix, weight=None):
        return F.linear(
            z,
            w[prefix + ".weight"] if weight is None else weight,
            w.get(prefix + ".bias"),
        )

    causal = torch.ones(length, length, dtype=torch.bool).tril()
    traces = []
    for block in program["blocks"]:
        i = block["layer"]
        a = block["attention"]
        m = block["mlp"]
        edits = [e for e in program["edits"] if e["layer"] == i]
        z = norm(x, block["norms"][0])

        def heads(v, n):
            return v.view(v.shape[0], length, n, head_dim).transpose(1, 2)

        q = heads(linear(z, a["q"]), c.num_attention_heads)
        k = heads(linear(z, a["k"]), c.num_key_value_heads)
        v = heads(linear(z, a["v"]), c.num_key_value_heads)
        q = q * cos + rotate(q) * sin
        k = k * cos + rotate(k) * sin
        for e in edits:
            if e["op"] == "attention_scale":
                q[:, e["head"]] *= e["factor"]
        repeat = c.num_attention_heads // c.num_key_value_heads
        k = k.repeat_interleave(repeat, 1)
        v = v.repeat_interleave(repeat, 1)
        scores = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
        scores = torch.where(causal, scores, torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            scores += (
                1 - attention_mask[:, None, None, :].to(scores.dtype)
            ) * torch.finfo(scores.dtype).min
        attention = scores.softmax(-1)
        out = attention @ v
        for e in edits:
            if e["op"] == "head_scale":
                out[:, e["head"]] *= e["factor"]
        x = x + linear(
            out.transpose(1, 2).contiguous().view(x.shape[0], length, -1), a["write"]
        )
        z = norm(x, block["norms"][1])
        hidden = ACT2FN[m["op"]](linear(z, m["gate"])) * linear(z, m["up"])
        effective = w[m["write"] + ".weight"].clone()
        for e in edits:
            if e["op"] == "neuron_scale":
                effective[:, e["neurons"]] *= e["factor"]
            elif e["op"] == "remove_direction":
                d = effective.new_tensor(e["direction"])
                d /= d.norm()
                columns = effective[:, e["neurons"]]
                effective[:, e["neurons"]] = (
                    columns - e["strength"] * d[:, None] * (d @ columns)[None, :]
                )
        x = x + linear(hidden, m["write"], effective)
        if capture:
            traces.append(
                {
                    "layer": i,
                    "attention": attention.cpu(),
                    "neurons": hidden.cpu(),
                    "residual": x.cpu(),
                }
            )
    logits = linear(norm(x, "model.norm"), "lm_head")
    return (logits, traces) if capture else logits


@torch.no_grad()
def compile(model, program):
    validate_program(model, program)
    result = copy.deepcopy(model).eval()
    c = model.config
    dim = getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)
    for e in program["edits"]:
        b = result.model.layers[e["layer"]]
        op = e["op"]
        if op == "head_scale":
            b.self_attn.o_proj.weight[:, e["head"] * dim : (e["head"] + 1) * dim] *= e[
                "factor"
            ]
        elif op == "attention_scale":
            sl = slice(e["head"] * dim, (e["head"] + 1) * dim)
            b.self_attn.q_proj.weight[sl] *= e["factor"]
            if b.self_attn.q_proj.bias is not None:
                b.self_attn.q_proj.bias[sl] *= e["factor"]
        elif op == "neuron_scale":
            b.mlp.down_proj.weight[:, e["neurons"]] *= e["factor"]
        elif op == "remove_direction":
            w = b.mlp.down_proj.weight
            d = w.new_tensor(e["direction"])
            d /= d.norm()
            cols = w[:, e["neurons"]]
            w[:, e["neurons"]] = cols - e["strength"] * d[:, None] * (d @ cols)[None, :]
    if any(not torch.isfinite(p).all() for p in result.parameters()):
        raise ValueError("nonfinite compiled weights")
    return result
