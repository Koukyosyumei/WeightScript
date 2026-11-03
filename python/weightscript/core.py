"""Independent executable GPT-2 computation IR and architecture-preserving compiler.

This extracts the numerical computation, not a symbolic explanation of its behavior.
Native LayerNorm, softmax and floating point are preserved; no linearization.
"""

import copy
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.activations import ACT2FN


def load_model(source):
    model = (
        AutoModelForCausalLM.from_pretrained(
            source,
            attn_implementation="eager",
            torch_dtype=torch.float32,
            trust_remote_code=False,
        )
        .cpu()
        .eval()
    )
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("expected float32 checkpoint")
    return model


def validate_model(model):
    c = model.config
    if c.model_type in {"llama", "qwen2"}:
        from .adapters import validate

        return validate(model)
    if c.model_type != "gpt2" or type(model).__name__ != "GPT2LMHeadModel":
        raise ValueError(
            "this adapter supports standard GPT2LMHeadModel checkpoints only"
        )
    if (
        c.reorder_and_upcast_attn
        or getattr(c, "add_cross_attention", False)
        or getattr(c, "pruned_heads", {})
    ):
        raise ValueError(
            "upcast attention, cross-attention and pruned-head checkpoints are not supported"
        )
    if any(
        p.dtype != torch.float32 or p.device.type != "cpu" for p in model.parameters()
    ):
        raise ValueError("adapter requires float32 CPU weights")
    if model.training:
        raise ValueError("model must be in eval mode")


def semantic_config(model):
    if model.config.model_type != "gpt2":
        return {
            k: v
            for k, v in model.config.to_dict().items()
            if not k.startswith("_")
            and k
            not in {"transformers_version", "architectures", "dtype", "torch_dtype"}
        }
    fields = (
        "n_embd",
        "n_head",
        "n_layer",
        "n_positions",
        "n_inner",
        "vocab_size",
        "activation_function",
        "layer_norm_epsilon",
        "scale_attn_weights",
        "scale_attn_by_inverse_layer_idx",
        "reorder_and_upcast_attn",
        "tie_word_embeddings",
        "add_cross_attention",
    )
    return {k: getattr(model.config, k, None) for k in fields}


def fingerprint(model):
    h = hashlib.sha256()
    # Hash inference semantics, not transient repository paths/version metadata.
    h.update(json.dumps(semantic_config(model), sort_keys=True).encode())
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def extract(model):
    if model.config.model_type in {"llama", "qwen2"}:
        from .adapters import extract as adapter_extract

        return adapter_extract(model)
    validate_model(model)
    c = model.config
    blocks = []
    for i in range(c.n_layer):
        prefix = f"transformer.h.{i}"
        blocks.append(
            {
                "layer": i,
                "attention": {
                    "op": "causal_softmax_attention",
                    "heads": c.n_head,
                    "qkv": prefix + ".attn.c_attn",
                    "write": prefix + ".attn.c_proj",
                },
                "mlp": {
                    "op": c.activation_function,
                    "read": prefix + ".mlp.c_fc",
                    "write": prefix + ".mlp.c_proj",
                },
                "norms": [prefix + ".ln_1", prefix + ".ln_2"],
            }
        )
    return {
        "format": "weightscript-gpt2-v1",
        "base_sha256": fingerprint(model),
        "semantics": "float32 inference; causal attention; exact LayerNorm; no KV cache",
        "blocks": blocks,
        "edits": [],
    }


def validate_program(model, program):
    if model.config.model_type in {"llama", "qwen2"}:
        from .adapters import validate_program as adapter_validate

        return adapter_validate(model, program)
    base = extract(model)
    for key in ("format", "base_sha256", "semantics", "blocks"):
        if program.get(key) != base[key]:
            raise ValueError(f"program {key} does not match checkpoint")
    if set(program) != set(base):
        raise ValueError("unknown program fields")
    c = model.config
    if not isinstance(program["edits"], list):
        raise ValueError("edits must be a list")
    for edit in program["edits"]:
        if not isinstance(edit, dict):
            raise ValueError("each edit must be an object")
        op = edit.get("op")
        required = {"op", "layer"}
        if op in ("head_scale", "attention_scale"):
            required |= {"head", "factor"}
        elif op == "neuron_scale":
            required |= {"neurons", "factor"}
        elif op == "remove_direction":
            required |= {"neurons", "direction", "strength"}
        else:
            raise ValueError(f"unsupported edit {op!r}")
        if set(edit) != required:
            raise ValueError(f"invalid fields for {op}")
        layer = edit["layer"]
        if type(layer) is not int or not 0 <= layer < c.n_layer:
            raise ValueError("layer out of bounds")
        if "head" in edit and (
            type(edit["head"]) is not int or not 0 <= edit["head"] < c.n_head
        ):
            raise ValueError("head out of bounds")
        if "neurons" in edit:
            units = edit["neurons"]
            width = model.transformer.h[layer].mlp.c_proj.weight.shape[0]
            if (
                not isinstance(units, list)
                or not units
                or any(type(n) is not int or not 0 <= n < width for n in units)
                or len(units) != len(set(units))
            ):
                raise ValueError("neurons must be unique in-range indices")
        if "direction" in edit:
            d = edit["direction"]
            if (
                not isinstance(d, list)
                or len(d) != c.n_embd
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in d)
                or sum(v * v for v in d) == 0
            ):
                raise ValueError("direction must be finite, nonzero and residual-width")
        factor = edit.get("factor", edit.get("strength"))
        if (
            not isinstance(factor, (int, float))
            or not math.isfinite(factor)
            or abs(factor) > 1e4
        ):
            raise ValueError("edit factor must be finite and at most 1e4 in magnitude")


def direction(edit, reference):
    d = reference.new_tensor(edit["direction"])
    norm = torch.linalg.vector_norm(d)
    if not torch.isfinite(d).all() or not torch.isfinite(norm) or norm <= 0:
        raise ValueError("direction cannot be normalized safely in float32")
    return d / norm


@torch.inference_mode()
def program_forward(
    model, program, input_ids, attention_mask=None, capture=False, validated=False
):
    if model.config.model_type in {"llama", "qwen2"}:
        from .adapters import forward

        return forward(model, program, input_ids, attention_mask, capture, validated)
    if not validated:
        validate_program(model, program)
    c = model.config
    if (
        input_ids.ndim != 2
        or input_ids.shape[1] < 1
        or input_ids.shape[1] > c.n_positions
    ):
        raise ValueError(
            "input must be a nonempty [batch, sequence] tensor within context length"
        )
    if input_ids.min() < 0 or input_ids.max() >= c.vocab_size:
        raise ValueError("input token outside vocabulary")
    w = model.state_dict()

    def norm(x, prefix):
        return F.layer_norm(
            x,
            (c.n_embd,),
            w[prefix + ".weight"],
            w[prefix + ".bias"],
            c.layer_norm_epsilon,
        )

    def linear(x, prefix):
        # HF GPT-2 Conv1D uses [in, out], not nn.Linear's [out, in].
        shape = x.shape[:-1] + (w[prefix + ".bias"].numel(),)
        return torch.addmm(
            w[prefix + ".bias"], x.reshape(-1, x.shape[-1]), w[prefix + ".weight"]
        ).view(shape)

    length = input_ids.shape[1]
    x = F.embedding(input_ids, w["transformer.wte.weight"]) + F.embedding(
        torch.arange(length, device=input_ids.device), w["transformer.wpe.weight"]
    )
    causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
    traces = []
    for block in program["blocks"]:
        i = block["layer"]
        edits = [e for e in program["edits"] if e["layer"] == i]
        a = block["attention"]
        q, k, v = linear(norm(x, block["norms"][0]), a["qkv"]).split(c.n_embd, dim=-1)

        def heads(z):
            return z.view(z.shape[0], length, c.n_head, c.n_embd // c.n_head).transpose(
                1, 2
            )

        q, k, v = heads(q), heads(k), heads(v)
        # Compiling attention_scale multiplies Q weights and Q bias.
        for e in edits:
            if e["op"] == "attention_scale":
                q[:, e["head"]] *= e["factor"]
        scores = q @ k.transpose(-1, -2)
        if c.scale_attn_weights:
            scores = scores / math.sqrt(v.shape[-1])
        if c.scale_attn_by_inverse_layer_idx:
            scores = scores / (i + 1)
        scores = torch.where(causal, scores, torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            scores = (
                scores
                + (1 - attention_mask[:, None, None, :].to(scores.dtype))
                * torch.finfo(scores.dtype).min
            )
        attention = F.softmax(scores, dim=-1).to(v.dtype)
        values = attention @ v
        for e in edits:
            if e["op"] == "head_scale":
                values[:, e["head"]] *= e["factor"]
        joined = values.transpose(1, 2).contiguous().view(x.shape)
        x = x + linear(joined, a["write"])
        m = block["mlp"]
        hidden = ACT2FN[m["op"]](linear(norm(x, block["norms"][1]), m["read"]))
        # Process operations in order on the effective write matrix. The IR's
        # edit semantics and compilation share algebra, but not model forward.
        effective = w[m["write"] + ".weight"]
        if any(e["op"] in ("neuron_scale", "remove_direction") for e in edits):
            effective = effective.clone()
            for e in edits:
                if e["op"] == "neuron_scale":
                    effective[e["neurons"]] *= e["factor"]
                elif e["op"] == "remove_direction":
                    d = direction(e, effective)
                    rows = effective[e["neurons"]]
                    effective[e["neurons"]] = (
                        rows - e["strength"] * (rows @ d).unsqueeze(1) * d
                    )
        x = x + (hidden @ effective + w[m["write"] + ".bias"])
        if capture:
            traces.append(
                {
                    "layer": i,
                    "attention": attention.cpu(),
                    "neurons": hidden.cpu(),
                    "residual": x.cpu(),
                }
            )
    logits = F.linear(norm(x, "transformer.ln_f"), w["lm_head.weight"])
    return (logits, traces) if capture else logits


@torch.no_grad()
def compile_edits(model, program):
    if model.config.model_type in {"llama", "qwen2"}:
        from .adapters import compile as adapter_compile

        return adapter_compile(model, program)
    validate_program(model, program)
    updated = copy.deepcopy(model).eval()
    c = model.config
    for edit in program["edits"]:
        block = updated.transformer.h[edit["layer"]]
        op = edit["op"]
        if op == "head_scale":
            d = c.n_embd // c.n_head
            rows = slice(edit["head"] * d, (edit["head"] + 1) * d)
            block.attn.c_proj.weight[rows] *= edit["factor"]
        elif op == "attention_scale":
            d = c.n_embd // c.n_head
            cols = slice(edit["head"] * d, (edit["head"] + 1) * d)
            block.attn.c_attn.weight[:, cols] *= edit["factor"]
            block.attn.c_attn.bias[cols] *= edit["factor"]
        elif op == "neuron_scale":
            block.mlp.c_proj.weight[edit["neurons"]] *= edit["factor"]
        elif op == "remove_direction":
            weight = block.mlp.c_proj.weight
            d = direction(edit, weight)
            rows = weight[edit["neurons"]]
            weight[edit["neurons"]] = (
                rows - edit["strength"] * (rows @ d).unsqueeze(1) * d
            )
    if any(not torch.isfinite(p).all() for p in updated.parameters()):
        raise ValueError("edit produced non-finite weights")
    return updated


def audit(before, after):
    a, b = before.state_dict(), after.state_dict()
    if a.keys() != b.keys() or any(a[k].shape != b[k].shape for k in a):
        raise ValueError("architecture changed")
    if semantic_config(before) != semantic_config(after):
        raise ValueError("architecture configuration changed")

    def alias_key(t):
        return (
            t.untyped_storage().data_ptr(),
            t.storage_offset(),
            tuple(t.shape),
            tuple(t.stride()),
        )

    def aliases(state):
        groups = {}
        for name, tensor in state.items():
            groups.setdefault(alias_key(tensor), []).append(name)
        return sorted(sorted(names) for names in groups.values())

    if aliases(a) != aliases(b):
        raise ValueError("weight tying changed")
    changes = []
    seen = set()
    for name in a:
        key = alias_key(a[name])
        if key in seen:
            continue
        seen.add(key)
        count = int(torch.count_nonzero(a[name] != b[name]))
        if count:
            changes.append(
                {
                    "tensor": name,
                    "changed": count,
                    "total": a[name].numel(),
                    "aliases": [
                        other
                        for other in a
                        if other != name and alias_key(a[other]) == key
                    ],
                    "delta_l2": float(torch.linalg.vector_norm(b[name] - a[name])),
                }
            )
    return {
        "architecture_preserved": True,
        "gradient_updates": 0,
        "changed_scalars": sum(row["changed"] for row in changes),
        "tensors": changes,
    }


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


@torch.no_grad()
def apply_raw_edits(model, edits):
    """Direct tensor editing baseline, without semantic layer/head/neuron names."""
    updated = copy.deepcopy(model).eval()
    weights = dict(updated.named_parameters())
    for e in edits:
        op = e.get("op")
        fields = {"op", "tensor", "rows"} | (
            {"factor"} if op == "scale" else {"direction", "strength"}
        )
        if op not in ("scale", "remove_direction") or set(e) != fields:
            raise ValueError("invalid raw tensor edit")
        if e["tensor"] not in weights:
            raise ValueError("unknown raw parameter name")
        w = weights[e["tensor"]]
        rows = e["rows"]
        if (
            w.ndim != 2
            or not rows
            or len(rows) != len(set(rows))
            or any(type(i) is not int or not 0 <= i < w.shape[0] for i in rows)
        ):
            raise ValueError("raw edits require unique in-range matrix rows")
        factor = e.get("factor", e.get("strength"))
        if (
            not isinstance(factor, (int, float))
            or not math.isfinite(factor)
            or abs(factor) > 1e4
        ):
            raise ValueError("invalid raw edit factor")
        if op == "scale":
            w[rows] *= factor
        else:
            d = w.new_tensor(e["direction"])
            if d.shape != (w.shape[1],) or not torch.isfinite(d).all() or d.norm() == 0:
                raise ValueError("invalid raw direction")
            d = d / d.norm()
            selected = w[rows]
            w[rows] = selected - factor * (selected @ d).unsqueeze(1) * d
    if any(not torch.isfinite(p).all() for p in updated.parameters()):
        raise ValueError("raw edit produced non-finite weights")
    return updated
