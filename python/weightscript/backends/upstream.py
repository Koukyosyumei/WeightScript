"""Adapter for the real lacoco-lab component/path pruning implementation.

Uses upstream hooks, masks and source renderer. Greedy ablation is an explicit
local driver, not a reproduction of their optimized three-stage paper pipeline.
"""

import copy
import importlib
import json
import subprocess
import sys
import types
from pathlib import Path

import torch

from ..core import fingerprint
from ..evaluate import batches
from ..ir import Artifact, Node

REVISION = "787398b0f4393a90ad02cd8e71be48d1e85d2b21"
URL = "https://github.com/lacoco-lab/decompiling_transformers"


def modules(checkout):
    root = Path(checkout).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"expected upstream revision {REVISION}, got {revision}")
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if status.strip():
        raise ValueError("upstream tracked source has local modifications")
    directory = root / "crasp/scripts/patching"
    sys.path.insert(0, str(directory))
    # Upstream's renderer imports a machine-local path file that is not shipped.
    # Supply only its documented directory constant; algorithm code is untouched.
    paths = types.ModuleType("my_paths")
    paths.path_to_saved_model = str(root / "share/saved_models")
    sys.modules["my_paths"] = paths
    try:
        pruning = importlib.import_module("pruning_model")
        helpers = importlib.import_module("patching_helper_functions")
        renderer = importlib.import_module("convert_to_code")
    finally:
        sys.path.remove(str(directory))
    return pruning, helpers, renderer


@torch.inference_mode()
def _engine(
    model, upstream, config, sampler, ln_variances, linear_ln, hooked_copy=None
):
    if hooked_copy is None:
        hooked_copy = copy.deepcopy(model).eval()
    oa = upstream.OptimalAblationVectors(
        sampler.input_vertex,
        sampler.output_vertex,
        sampler.output_vertex,
        None,
        model.config,
        ln_variances,
    )
    # Preserve attention output biases which upstream stores separately from heads.
    for layer in range(model.config.n_layer):
        for head in range(model.config.n_head):
            for act in ("q", "k", "v"):
                for previous in range(layer):
                    oa.output_vertex_oa[oa.to_out_oa_idx[(layer, act, head)]] += (
                        hooked_copy.transformer.h[previous].attn.c_proj.bias
                    )
        for previous in range(layer + 1):
            oa.output_vertex_oa[oa.to_out_oa_idx[(layer, "mlp")]] += (
                hooked_copy.transformer.h[previous].attn.c_proj.bias
            )
    for layer in range(model.config.n_layer):
        oa.output_vertex_oa[oa.to_out_oa_idx[("lm_head",)]] += (
            hooked_copy.transformer.h[layer].attn.c_proj.bias
        )
    hooked = upstream.PruningModelWithHooks(
        hooked_copy,
        config,
        sampler.mapping_to_param_idx,
        logger=lambda *a: None,
        linear_LN=linear_ln,
    )
    return hooked, oa


@torch.inference_mode()
def decompile(
    model,
    checkout,
    rows,
    max_trials=64,
    min_fidelity=0.99,
    batch_size=32,
    linear_ln=False,
    evaluation_rows=None,
):
    if model.config.model_type != "gpt2":
        raise ValueError("upstream backend supports GPT-2 only")
    if not rows or max_trials < 0 or not 0 <= min_fidelity <= 1:
        raise ValueError("invalid pruning domain or budget")
    upstream, helpers, renderer = modules(checkout)
    original_hash = fingerprint(model)
    native_batches = list(batches(rows, batch_size))
    native = [model(ids, use_cache=False).logits.detach() for _, ids in native_batches]
    config = helpers.get_full_possible_config_for_pruning(
        {i: model.config.n_head for i in range(model.config.n_layer)}
    )
    sampler = upstream.MaskSampler(config)
    hooked_copy = copy.deepcopy(model).eval()
    ln_variances = torch.zeros(len(sampler.output_vertex))
    # For linearization, estimate per-layer variance from the calibration domain.
    if linear_ln:
        variances = {}
        hooks = []
        norm_modules = {("final",): hooked_copy.transformer.ln_f}
        for i, block in enumerate(hooked_copy.transformer.h):
            norm_modules[(i, "qkv")] = block.ln_1
            norm_modules[(i, "mlp")] = block.ln_2
        for key, module in norm_modules.items():
            variances[key] = []
            hooks.append(
                module.register_forward_pre_hook(
                    lambda _, args, key=key: variances[key].append(
                        args[0].var(-1, unbiased=False).mean().item()
                    )
                )
            )
        try:
            for _, ids in native_batches:
                hooked_copy(ids, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        for i, vertex in enumerate(sampler.output_vertex):
            key = (
                ("final",)
                if vertex == ("lm_head",)
                else (vertex[0], "mlp" if vertex[1] == "mlp" else "qkv")
            )
            ln_variances[i] = (
                torch.tensor(sum(variances[key]) / len(variances[key]))
                .clamp_min(1e-12)
                .log()
            )
    hooked, oa = _engine(
        model, upstream, config, sampler, ln_variances, linear_ln, hooked_copy
    )
    masks = torch.ones(len(sampler.mapping_to_param_idx))
    forward_calls = 0

    def measure(mask, items=native_batches, targets=native):
        nonlocal forward_calls
        matches = total = 0
        error = 0.0
        for (_, ids), target in zip(items, targets):
            logits = hooked(
                ids, masks=mask.expand(len(ids), -1), oa_vecs=oa, use_cache=False
            ).logits
            forward_calls += 1
            # Match the final decision; report this domain explicitly.
            matches += int((logits[:, -1].argmax(-1) == target[:, -1].argmax(-1)).sum())
            total += len(ids)
            error = max(error, float((logits - target).abs().max()))
        return {
            "final_token_agreement": matches / total,
            "max_logit_error": error,
            "examples": total,
        }

    try:
        initial = measure(masks)
        trials = []
        # Ties and iteration order deterministic; use actual upstream path masks.
        for edge, index in sorted(
            sampler.mapping_to_param_idx.items(), key=lambda pair: str(pair[0])
        )[:max_trials]:
            trial = masks.clone()
            trial[index] = 0
            metric = measure(trial)
            accepted = metric["final_token_agreement"] >= min_fidelity
            if accepted:
                masks = trial
            trials.append({"edge": list(edge), "accepted": accepted, "metrics": metric})
        final = measure(masks)
        validation = None
        if evaluation_rows:
            validation_batches = list(batches(evaluation_rows, batch_size))
            # Native model has no hooks: evaluation cannot leak into pruning.
            validation_targets = [
                model(ids, use_cache=False).logits for _, ids in validation_batches
            ]
            validation = measure(masks, validation_batches, validation_targets)
        pruned = copy.deepcopy(config)
        upstream.convert_mask_to_config_(masks, pruned, sampler.mapping_to_param_idx)
        serial = json.loads(json.dumps(pruned))
        source = renderer.convert_config_to_code_one_step_paths(serial)
    finally:
        hooked.remove_hooks()
    if fingerprint(model) != original_hash:
        raise RuntimeError("upstream extraction mutated source model")
    nodes = [
        Node("tokens", "tokens", interpretation="primitive"),
        Node("positions", "indices", interpretation="primitive"),
    ]
    for layer in range(model.config.n_layer):
        for head in range(model.config.n_head):
            node_id = f"attention.{layer}.{head}"
            nodes.append(
                Node(
                    node_id,
                    "upstream_attention_paths",
                    ["tokens", "positions"],
                    {"paths": serial[str(layer)], "head": head},
                    [
                        f"transformer.h.{layer}.attn.c_attn.weight",
                        f"transformer.h.{layer}.attn.c_proj.weight",
                    ],
                    "opaque",
                )
            )
    nodes.append(
        Node(
            "output",
            "upstream_pruned_model",
            [n.id for n in nodes],
            {
                "source": source,
                "config": serial,
                "edge_masks": [
                    {"path": list(edge), "value": int(masks[index])}
                    for edge, index in sampler.mapping_to_param_idx.items()
                ],
                "normalization": [
                    {"vertex": list(vertex), "log_variance": float(ln_variances[index])}
                    for index, vertex in enumerate(sampler.output_vertex)
                ],
                "linear_ln": linear_ln,
            },
            interpretation="opaque",
        )
    )
    artifact = Artifact(
        "decompiling_transformers.stage1",
        nodes,
        "output",
        {
            "repository": URL,
            "revision": REVISION,
            "checkpoint_sha256": original_hash,
            "driver": "greedy zero-ablation; upstream mask/hook/renderer implementations",
        },
        {
            "claim": "empirical final-token fidelity on specified domains; no paper-wide reproduction claim",
            "layernorm": "calibrated linearization"
            if linear_ln
            else "native LayerNorm",
            "initial": initial,
            "calibration": final,
            "evaluation": validation,
            "component_edges_before": len(masks),
            "component_edges_after": int(masks.sum()),
            "native_forward_calls": len(native_batches)
            + (len(validation_batches) if evaluation_rows else 0)
            + (len(native_batches) if linear_ln else 0),
            "pruned_forward_calls": forward_calls,
            "trials": trials,
        },
    )
    return artifact


@torch.inference_mode()
def execute_artifact(model, artifact, checkout, input_ids):
    """Replay the serialized extracted path program, without rerunning pruning."""
    data = artifact.to_dict() if isinstance(artifact, Artifact) else artifact
    if (
        data.get("backend") != "decompiling_transformers.stage1"
        or data.get("version") != "weightscript.ir.v1"
    ):
        raise ValueError("unsupported upstream artifact")
    if data["provenance"]["checkpoint_sha256"] != fingerprint(model):
        raise ValueError("extracted program belongs to a different checkpoint")
    upstream, helpers, _ = modules(checkout)
    config = helpers.get_full_possible_config_for_pruning(
        {i: model.config.n_head for i in range(model.config.n_layer)}
    )
    sampler = upstream.MaskSampler(config)
    attrs = data["nodes"][-1]["attributes"]
    weights = {tuple(row["path"]): row["value"] for row in attrs["edge_masks"]}
    if set(weights) != set(sampler.mapping_to_param_idx) or any(
        v not in (0, 1) for v in weights.values()
    ):
        raise ValueError("invalid extracted edge masks")
    normalizations = {
        tuple(row["vertex"]): row["log_variance"] for row in attrs["normalization"]
    }
    values = torch.tensor([normalizations[vertex] for vertex in sampler.output_vertex])
    if not torch.isfinite(values).all() or values.abs().max() > 80:
        raise ValueError("invalid normalization values")
    hooked, oa = _engine(model, upstream, config, sampler, values, attrs["linear_ln"])
    mask = torch.tensor(
        [
            weights[edge]
            for edge, _ in sorted(
                sampler.mapping_to_param_idx.items(), key=lambda pair: pair[1]
            )
        ],
        dtype=torch.float32,
    )
    try:
        return hooked(
            input_ids,
            masks=mask.expand(len(input_ids), -1),
            oa_vecs=oa,
            use_cache=False,
        ).logits
    finally:
        hooked.remove_hooks()
