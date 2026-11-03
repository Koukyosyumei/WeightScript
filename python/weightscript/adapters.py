"""Independent full-sequence dense rotary decoder execution (no native forward)."""

import math

import torch
import torch.nn.functional as F
from transformers.activations import ACT2FN

CAPABILITIES = {
    "llama": {
        "class": "LlamaForCausalLM",
        "rope": {"default", "linear", "dynamic", "llama3"},
    },
    "qwen2": {
        "class": "Qwen2ForCausalLM",
        "rope": {"default", "linear", "dynamic", "llama3"},
    },
    "mistral": {
        "class": "MistralForCausalLM",
        "rope": {"default", "linear", "dynamic", "llama3"},
    },
}
SUPPORTED = set(CAPABILITIES)


def validate(model):
    c = model.config
    capability = CAPABILITIES.get(c.model_type)
    if capability is None:
        raise ValueError(
            f"unsupported decoder architecture: {c.model_type}; no semantic adapter"
        )
    if type(model).__name__ != capability["class"]:
        raise ValueError("custom decoder subclasses are unsupported")
    scaling = getattr(c, "rope_scaling", None) or {}
    if not isinstance(scaling, dict):
        raise ValueError("RoPE scaling must be an object")
    kind = scaling.get("rope_type", scaling.get("type", "default"))
    if kind not in capability["rope"]:
        raise ValueError(f"unsupported RoPE variant: {kind}")

    def positive(value):
        return type(value) in (int, float) and math.isfinite(value) and value > 0

    if not positive(c.rope_theta) or (
        kind != "default" and not positive(scaling.get("factor"))
    ):
        raise ValueError("invalid RoPE base or factor")
    if kind == "llama3" and (
        not all(
            positive(scaling.get(k))
            for k in (
                "original_max_position_embeddings",
                "low_freq_factor",
                "high_freq_factor",
            )
        )
        or scaling["low_freq_factor"] >= scaling["high_freq_factor"]
    ):
        raise ValueError("invalid llama3 RoPE frequency bounds")
    if c.hidden_act not in ACT2FN:
        raise ValueError("unsupported decoder activation")
    if getattr(c, "partial_rotary_factor", 1.0) != 1.0:
        raise ValueError("partial rotary dimensions unsupported")
    dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    if (
        dim <= 0
        or dim % 2
        or c.num_key_value_heads <= 0
        or c.num_attention_heads % c.num_key_value_heads
    ):
        raise ValueError("invalid rotary or grouped-query dimensions")
    if kind == "dynamic" and dim <= 2:
        raise ValueError("dynamic RoPE requires head dimension greater than two")
    params = list(model.parameters())
    if (
        model.training
        or not params
        or any(
            p.dtype not in {torch.float32, torch.float16, torch.bfloat16}
            or p.device.type not in {"cpu", "cuda"}
            or p.device != params[0].device
            or p.dtype != params[0].dtype
            for p in params
        )
    ):
        raise ValueError(
            "adapter requires eval-mode homogeneous floating CPU/CUDA weights"
        )
    if getattr(model, "is_quantized", False):
        raise ValueError("quantized weights are unsupported")
    layers = getattr(c, "layer_types", None)
    if layers is not None and (
        len(layers) != c.num_hidden_layers
        or any(t not in {"full_attention", "sliding_attention"} for t in layers)
    ):
        raise ValueError("unsupported attention layer types")
    if (
        layers
        and "sliding_attention" in layers
        and not getattr(c, "sliding_window", None)
    ):
        raise ValueError("sliding attention requires a positive window")
    window = getattr(c, "sliding_window", None)
    if c.model_type == "llama" and (
        window is not None or (layers and "sliding_attention" in layers)
    ):
        raise ValueError("Llama native attention does not implement sliding windows")
    if window is not None and (type(window) is not int or window <= 0):
        raise ValueError("invalid sliding window")


def rotary(c, length, dim, device, dtype):
    """Compute rotary frequencies independently of Transformers model helpers."""
    scaling = getattr(c, "rope_scaling", None) or {}
    kind = scaling.get("rope_type", scaling.get("type", "default"))
    base = c.rope_theta
    if kind == "dynamic" and length > c.max_position_embeddings:
        factor = scaling["factor"]
        base *= (factor * length / c.max_position_embeddings - (factor - 1)) ** (
            dim / (dim - 2)
        )
    inv = 1.0 / (
        base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    if kind == "linear":
        inv /= scaling["factor"]
    elif kind == "llama3":
        wavelength = 2 * math.pi / inv
        original = scaling["original_max_position_embeddings"]
        low, high = scaling["low_freq_factor"], scaling["high_freq_factor"]
        smooth = (original / wavelength - low) / (high - low)
        interpolated = (1 - smooth) * inv / scaling["factor"] + smooth * inv
        inv = torch.where(
            wavelength > original / low,
            inv / scaling["factor"],
            torch.where(wavelength < original / high, inv, interpolated),
        )
    angles = (
        torch.arange(length, device=device, dtype=torch.float32)[:, None] * inv[None, :]
    )
    angles = torch.cat((angles, angles), -1)
    return angles.cos().to(dtype)[None, None], angles.sin().to(dtype)[None, None]


def layer_window(c, index):
    types = getattr(c, "layer_types", None)
    if types:
        return c.sliding_window if types[index] == "sliding_attention" else None
    if c.model_type == "mistral" or getattr(c, "use_sliding_window", False):
        return getattr(c, "sliding_window", None)
    return None


def extract(model):
    from .core import fingerprint

    validate(model)
    c = model.config
    return {
        "format": f"weightscript-{c.model_type}-v1",
        "base_sha256": fingerprint(model),
        "semantics": (
            "float32 causal inference; RMSNorm; default RoPE; GQA; no KV cache"
            if next(model.parameters()).dtype == torch.float32
            and not getattr(c, "rope_scaling", None)
            and not any(layer_window(c, i) for i in range(c.num_hidden_layers))
            else "causal inference; float32-accumulating RMSNorm/softmax; configured RoPE/windows; GQA; no KV cache"
        ),
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
    if not isinstance(program, dict):
        raise ValueError("program must be an object")
    if set(program) != set(base) or any(
        program[k] != base[k] for k in base if k != "edits"
    ):
        raise ValueError("program metadata does not match decoder checkpoint")
    if not isinstance(program["edits"], list):
        raise ValueError("edits must be a list")
    for e in program["edits"]:
        if not isinstance(e, dict):
            raise ValueError("each edit must be an object")
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
            or any(
                type(n) is not int or not 0 <= n < model.config.intermediate_size
                for n in e["neurons"]
            )
            or len(set(e["neurons"])) != len(e["neurons"])
        ):
            raise ValueError("invalid neuron indices")
        f = e.get("factor", e.get("strength"))
        if type(f) not in (int, float) or not math.isfinite(f) or abs(f) > 1e4:
            raise ValueError("invalid factor")
        if op == "remove_direction":
            values = e["direction"]
            if not isinstance(values, list) or any(
                type(v) not in (int, float) for v in values
            ):
                raise ValueError("invalid residual direction")
            d = torch.tensor(values, dtype=torch.float32)
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
        not isinstance(ids, torch.Tensor)
        or ids.dtype not in {torch.int32, torch.int64}
        or ids.device != next(model.parameters()).device
        or ids.ndim != 2
        or not ids.numel()
        or ids.min() < 0
        or ids.max() >= c.vocab_size
    ):
        raise ValueError("invalid token domain")
    w = model.state_dict()
    x = F.embedding(ids, w["model.embed_tokens.weight"])
    length = ids.shape[1]
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    cos, sin = rotary(c, length, head_dim, x.device, x.dtype)
    if attention_mask is not None and (
        not isinstance(attention_mask, torch.Tensor)
        or attention_mask.shape != ids.shape
        or attention_mask.device != ids.device
        or not ((attention_mask == 0) | (attention_mask == 1)).all()
    ):
        raise ValueError("attention mask must be binary and match tokens/device")

    def rotate(z):
        first, second = z.chunk(2, -1)
        return torch.cat((-second, first), -1)

    def norm(z, prefix):
        normalized = z.float() * torch.rsqrt(
            z.float().pow(2).mean(-1, keepdim=True) + c.rms_norm_eps
        )
        return normalized.to(z.dtype) * w[prefix + ".weight"]

    def linear(z, prefix, weight=None):
        return F.linear(
            z,
            w[prefix + ".weight"] if weight is None else weight,
            w.get(prefix + ".bias"),
        )

    causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
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
        allowed = causal
        window = layer_window(c, i)
        if window:
            positions = torch.arange(length, device=x.device)
            allowed = allowed & (positions[:, None] - positions[None, :] < window)
        if attention_mask is not None:
            allowed = allowed[None, None] & attention_mask[:, None, None, :].bool()
        scores = torch.where(allowed, scores, torch.finfo(scores.dtype).min)
        attention = scores.softmax(-1, dtype=torch.float32).to(q.dtype)
        out = attention @ v
        for e in edits:
            if e["op"] == "head_scale":
                out[:, e["head"]] *= e["factor"]
        x = x + linear(
            out.transpose(1, 2).contiguous().view(x.shape[0], length, -1), a["write"]
        )
        z = norm(x, block["norms"][1])
        hidden = ACT2FN[m["op"]](linear(z, m["gate"])) * linear(z, m["up"])
        effective = w[m["write"] + ".weight"]
        if any(e["op"] in {"neuron_scale", "remove_direction"} for e in edits):
            effective = effective.clone()
        for e in edits:
            if e["op"] == "neuron_scale":
                effective[:, e["neurons"]] *= e["factor"]
            elif e["op"] == "remove_direction":
                d = torch.tensor(
                    e["direction"], device=effective.device, dtype=torch.float32
                )
                d = (d / d.norm()).to(effective.dtype)
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
def compile(model, program, *, share_untouched=False):
    validate_program(model, program)
    from .raw import isolated_copy

    touched = []
    for edit in program["edits"]:
        prefix = f"model.layers.{edit['layer']}."
        suffix = {
            "head_scale": "self_attn.o_proj",
            "attention_scale": "self_attn.q_proj",
            "neuron_scale": "mlp.down_proj",
            "remove_direction": "mlp.down_proj",
        }[edit["op"]]
        touched.append(prefix + suffix + ".weight")
        if (
            edit["op"] == "attention_scale"
            and prefix + suffix + ".bias" in model.state_dict()
        ):
            touched.append(prefix + suffix + ".bias")
    result = isolated_copy(model, touched, share_untouched)
    c = model.config
    dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
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
            d = torch.tensor(e["direction"], device=w.device, dtype=torch.float32)
            d = (d / d.norm()).to(w.dtype)
            cols = w[:, e["neurons"]]
            w[:, e["neurons"]] = cols - e["strength"] * d[:, None] * (d @ cols)[None, :]
    if any(not torch.isfinite(p).all() for p in result.parameters()):
        raise ValueError("nonfinite compiled weights")
    return result
