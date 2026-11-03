"""Bounded local driver of the pinned upstream component/full-path/QK engines.

Uses unchanged upstream conversions, hook engines and masks. Greedy zero-ablation
and bounded auxiliary-vector calibration replace the paper's cluster optimizer;
this is an integration, not a reproduction of its reported results.
"""

import copy
import json

import torch
import torch.nn.functional as F

from ..core import fingerprint
from ..evaluate import batches
from ..ir import Artifact, Node
from .upstream import REVISION, URL, _engine, decompile, modules


def _tuple(value):
    return tuple(_tuple(v) for v in value) if isinstance(value, (list, tuple)) else value


def _config(value):
    if isinstance(value, dict):
        return {int(k) if str(k).isdigit() else k: _config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_config(v) for v in value]
    return value


def _state(oa):
    fields = [("output_vertex", "output_vertex_oa"), ("LN_vertex", "LN_var")]
    if hasattr(oa, "input_vertex"):
        fields.append(("input_vertex", "input_vertex_oa"))
    if hasattr(oa, "key_names"):
        fields.append(("key_names", "q_bias_term"))
    return {param: [{"vertex": vertex, "value": values.detach().tolist()}
                    for vertex, values in zip(getattr(oa, vertices), getattr(oa, param))]
            for vertices, param in fields}


def _load_state(oa, state):
    fields = {"input_vertex_oa": "input_vertex", "output_vertex_oa": "output_vertex",
              "LN_var": "LN_vertex", "q_bias_term": "key_names"}
    with torch.no_grad():
        for param, rows in state.items():
            tensor = getattr(oa, param)
            values = {_tuple(row["vertex"]): row["value"] for row in rows}
            vertices = getattr(oa, fields[param])
            if len(values) != len(rows) or set(values) != set(vertices):
                raise ValueError("invalid auxiliary vector vertices")
            data = torch.tensor([values[v] for v in vertices], dtype=tensor.dtype)
            if tensor.numel() == 0 and data.numel() == 0:
                data = data.reshape(tensor.shape)
            if data.shape != tensor.shape or not torch.isfinite(data).all():
                raise ValueError("invalid auxiliary vector values")
            if param == "LN_var" and data.abs().max() > 80:
                raise ValueError("invalid normalization values")
            tensor.copy_(data)


def _qk_batch_compatibility(hooked):
    # HF emits positional embeddings with a singleton batch axis. Upstream QK
    # directly views a selected V path as a full batch; broadcast its cached
    # position activation explicitly, without changing any upstream source.
    def expand_positions(module, args, output):
        batch = hooked.activations["wte"].shape[0]
        hooked.activations["wpe"] = output.expand(batch, -1, -1)
    hooked.hooks.append(hooked.model.transformer.wpe.register_forward_hook(expand_positions))


def _measure(hooked, oa, mask, items, targets):
    correct = total = all_correct = positions = 0
    error = 0.0
    counterexamples = []
    with torch.no_grad():
        for (rows, ids), target in zip(items, targets):
            actual = hooked(ids, masks=mask.expand(len(ids), -1), oa_vecs=oa, use_cache=False).logits
            if not torch.isfinite(actual).all() or not torch.isfinite(target).all():
                raise ValueError("nonfinite stage logits")
            match = actual[:, -1].argmax(-1) == target[:, -1].argmax(-1)
            correct += int(match.sum())
            total += len(ids)
            all_correct += int((actual.argmax(-1) == target.argmax(-1)).sum())
            positions += ids.numel()
            error = max(error, float((actual - target).abs().max()))
            for row, good, pred, expected in zip(rows, match, actual[:, -1].argmax(-1), target[:, -1].argmax(-1)):
                if not good and len(counterexamples) < 20:
                    counterexamples.append({"input_ids": row["input_ids"], "actual": int(pred), "expected": int(expected)})
    return {"final_token_agreement": correct / total, "all_token_agreement": all_correct / positions,
            "max_logit_error": error, "examples": total, "counterexamples": counterexamples}


def _fit(hooked, oa, mask, items, targets, steps, learning_rate):
    oa.requires_grad_(False)
    parameters = [oa.output_vertex_oa]
    if hasattr(oa, "q_bias_term"):
        parameters.append(oa.q_bias_term)
    for param in parameters:
        param.requires_grad_(True)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    losses = []
    with torch.enable_grad():
        for step in range(steps):
            (_, ids), target = items[step % len(items)], targets[step % len(targets)]
            optimizer.zero_grad()
            actual = hooked(ids, masks=mask.expand(len(ids), -1), oa_vecs=oa, use_cache=False).logits
            loss = F.kl_div(actual.log_softmax(-1), target.softmax(-1), reduction="batchmean") / ids.shape[1]
            if not torch.isfinite(loss):
                raise ValueError("nonfinite auxiliary calibration loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    oa.requires_grad_(False)
    return losses


def decompile_stages(model, checkout, rows, evaluation_rows=None, *, max_trials=32,
                     calibration_steps=32, learning_rate=0.01, min_fidelity=0.99,
                     batch_size=32):
    """Execute all three engines and serialize masks, auxiliary vectors and code.

    Every stage is compared to the frozen native model, including failed
    normalization assumptions. A low-fidelity stage remains opaque and is
    explicitly rejected, while its executable evidence is retained.
    """
    if (type(max_trials) is not int or max_trials < 0 or type(calibration_steps) is not int
            or not 0 <= calibration_steps <= 10000 or not 0 < learning_rate <= 1
            or not 0 <= min_fidelity <= 1):
        raise ValueError("invalid stage driver budget/configuration")
    # Includes model/domain checks and independent normalization calibration.
    component = decompile(model, checkout, rows, max_trials=max_trials,
                          min_fidelity=min_fidelity, batch_size=batch_size, linear_ln=True,
                          evaluation_rows=evaluation_rows, require_initial_fidelity=False)
    upstream, _, renderer = modules(checkout)
    source_hash = fingerprint(model)
    items = list(batches(rows, batch_size))
    tests = list(batches(evaluation_rows, batch_size)) if evaluation_rows else []
    with torch.no_grad():
        targets = [model(ids, use_cache=False).logits.detach() for _, ids in items]
        test_targets = [model(ids, use_cache=False).logits.detach() for _, ids in tests]
    attrs = component.nodes[-1].attributes
    config = _config(attrs["config"])
    heads = {i: model.config.n_head for i in range(model.config.n_layer)}
    # Recover component OA constants and calibrated variances in the full graph.
    full = upstream.MaskSampler(copy.deepcopy(config))
    # _engine's constants depend only on architecture, not the pruned edge set.
    variances = {_tuple(r["vertex"]): r["log_variance"] for r in attrs["normalization"]}
    hooked, previous_oa = _engine(model, upstream, config, full,
                                torch.tensor([variances[v] for v in full.output_vertex]), True)
    hooked.remove_hooks()
    stages = [{"name": "component", "artifact": component.to_dict(),
               "accepted": component.fidelity["calibration"]["final_token_agreement"] >= min_fidelity
               and (not evaluation_rows or component.fidelity["evaluation"]["final_token_agreement"] >= min_fidelity)}]
    calls = 0
    for name in ("full_path", "qk"):
        if name == "full_path":
            upstream.convert_config_fullpaths_(config, heads)
            sampler = upstream.MaskSamplerFullPaths(config, False)
            values = []
            for vertex in sampler.all_output_vertex:
                original = vertex[:3] if len(vertex) == 4 else vertex
                values.append(previous_oa.LN_var[previous_oa.to_LN_idx[original]].detach())
            oa = upstream.OptimalAblationVectors(sampler.input_vertex, sampler.output_vertex,
                                               sampler.all_output_vertex, None, model.config,
                                               torch.stack(values))
            with torch.no_grad():
                for v in oa.output_vertex:
                    if v in previous_oa.to_out_oa_idx:
                        oa.output_vertex_oa[oa.to_out_oa_idx[v]].copy_(previous_oa.output_vertex_oa[previous_oa.to_out_oa_idx[v]])
            hooked = upstream.PruningModelWithHooksFullPaths(copy.deepcopy(model), config,
                                                            sampler.mapping_to_param_idx, False, lambda *a: None)
        else:
            upstream.convert_full_paths_config_to_prune_inside_kq_config_(config, heads)
            sampler = upstream.MaskSamplerForQK(config)
            oa = upstream.OptimalQueryBiasVectors(sampler.key_names,
                                                 model.config.n_embd // model.config.n_head, previous_oa)
            # Exact affine query term from the preceding full-path engine.
            with torch.no_grad():
                for layer, head, key in sampler.key_names:
                    attn = model.transformer.h[layer].attn
                    width = attn.head_dim
                    start = head * width
                    v = previous_oa.output_vertex_oa[previous_oa.to_out_oa_idx[(layer, "q", head)]]
                    norm = model.transformer.h[layer].ln_1
                    variance = previous_oa.LN_var[previous_oa.to_LN_idx[(layer, "q", head)]].exp()
                    normalized = (v - v.mean()) / (variance + norm.eps).sqrt() * norm.weight
                    oa.q_bias_term[oa.to_q_bias[(layer, head, key)]].copy_(
                        normalized @ attn.c_attn.weight[:, start:start+width] + attn.c_attn.bias[start:start+width])
            hooked = upstream.PruningModelWithHooksForQK(copy.deepcopy(model), config,
                                                        sampler.mapping_to_param_idx, False, lambda *a: None)
            _qk_batch_compatibility(hooked)
        engine_config = copy.deepcopy(config)
        mask = torch.ones(len(sampler.mapping_to_param_idx))
        try:
            initial = _measure(hooked, oa, mask, items, targets)
            losses = _fit(hooked, oa, mask, items, targets, calibration_steps, learning_rate)
            calibrated = _measure(hooked, oa, mask, items, targets)
            trials = []
            for edge, index in sorted(sampler.mapping_to_param_idx.items(), key=lambda p: str(p[0]))[:max_trials]:
                trial = mask.clone()
                trial[index] = 0
                metric = _measure(hooked, oa, trial, items, targets)
                accepted = metric["final_token_agreement"] >= min_fidelity
                if accepted:
                    mask = trial
                trials.append({"edge": edge, "accepted": accepted, "metrics": metric})
            final = _measure(hooked, oa, mask, items, targets)
            validation = _measure(hooked, oa, mask, tests, test_targets) if tests else None
            calls += (3 + len(trials)) * len(items) + calibration_steps + len(tests)
            converter = upstream.convert_mask_to_config_qk_ if name == "qk" else upstream.convert_mask_to_config_
            converter(mask, config, sampler.mapping_to_param_idx)
            serial = json.loads(json.dumps(config))
            render_warning = None
            if name == "qk":
                source = renderer.convert_config_to_code_qk_pruned(serial, False, None)
            else:
                try:
                    source = renderer.convert_config_to_code_full_paths(serial)
                except UnboundLocalError as exc:
                    if "labels" not in str(exc):
                        raise
                    # Pinned renderer references heatmap labels even when plotting
                    # is off. Use its unchanged QK renderer on an expanded COPY.
                    display_config = copy.deepcopy(config)
                    upstream.convert_full_paths_config_to_prune_inside_kq_config_(display_config, heads)
                    source = renderer.convert_config_to_code_qk_pruned(json.loads(json.dumps(display_config)), False, None)
                    render_warning = "upstream full-path renderer has undefined heatmap labels; rendered equivalent expanded config with upstream QK renderer"
            stages.append({"name": name, "engine_config": engine_config, "config": serial,
                           "auxiliary": _state(oa), "source": source, "renderer_adaptation": render_warning,
                           "masks": [{"edge": edge, "value": int(mask[index])}
                                     for edge, index in sampler.mapping_to_param_idx.items()],
                           "initial": initial, "calibrated": calibrated, "calibration": final,
                           "evaluation": validation, "calibration_losses": losses,
                           "trials": trials, "edges_before": len(mask), "edges_after": int(mask.sum()),
                           "accepted": final["final_token_agreement"] >= min_fidelity
                           and (validation is None or validation["final_token_agreement"] >= min_fidelity)})
        finally:
            hooked.remove_hooks()
        previous_oa = oa
    if fingerprint(model) != source_hash:
        raise RuntimeError("upstream stages mutated frozen source weights")
    adaptations = ["broadcast cached positional embeddings across QK batches"]
    if any(stage.get("renderer_adaptation") for stage in stages):
        adaptations.append("QK renderer fallback for full-path renderer undefined heatmap labels")
    artifact = Artifact("decompiling_transformers.three_stage", [
        Node("tokens", "tokens", interpretation="primitive"),
        Node("output", "upstream_three_stage", ["tokens"], {"stages": stages},
             [n for n, _ in model.named_parameters()], "opaque")], "output",
        {"repository": URL, "revision": REVISION, "checkpoint_sha256": source_hash,
         "driver": "bounded local greedy masks and auxiliary-vector KL calibration; unchanged upstream engines",
         "completed_stages": [s["name"] for s in stages],
         "source_weights_frozen": True, "split_mlps": False,
         "compatibility_adaptations": adaptations},
        {"claim": "measured domain fidelity; stage execution does not establish paper reproduction or symbolic equivalence",
         "normalization": {
             "positions": "implicit positions 0..length-1 on the caller-supplied token ids",
             "layernorm": "upstream engines consume per-vertex log-variance fit on the calibration rows",
             "acceptance": "final-token argmax agreement; max_logit_error is separate and can reject a logit-level claim",
         },
         "accepted": all(s["accepted"] for s in stages),
         "pruned_forward_calls_after_component": calls,
         "native_forward_calls_after_component": len(items) + len(tests),
         "component": component.fidelity})
    from .upstream_primitives import match_qk_primitives
    match_qk_primitives(model, artifact, checkout)
    return artifact


@torch.no_grad()
def execute_stages(model, artifact, checkout, input_ids, stage="qk"):
    """Replay a serialized full-path or QK artifact; no optimization or data needed."""
    data = artifact.to_dict() if isinstance(artifact, Artifact) else artifact
    if (data.get("backend") != "decompiling_transformers.three_stage"
            or data.get("version") != "weightscript.ir.v1"
            or data.get("provenance", {}).get("checkpoint_sha256") != fingerprint(model)):
        raise ValueError("unsupported or checkpoint-mismatched stage artifact")
    if stage not in ("full_path", "qk"):
        raise ValueError("replay stage must be full_path or qk")
    entry = next(s for s in data["nodes"][-1]["attributes"]["stages"] if s["name"] == stage)
    upstream, _, _ = modules(checkout)
    config = _config(entry["engine_config"])
    if stage == "qk":
        for layer in range(model.config.n_layer):
            for head in config[layer]["qk"]:
                config[layer]["qk"][head] = [_tuple(p) for p in config[layer]["qk"][head]]
        sampler = upstream.MaskSamplerForQK(config)
    else:
        sampler = upstream.MaskSamplerFullPaths(config, False)
    state = entry["auxiliary"]
    vertices = lambda field: [_tuple(row["vertex"]) for row in state[field]]
    base = upstream.OptimalAblationVectors(
        vertices("input_vertex_oa") if stage == "full_path" else [], vertices("output_vertex_oa"),
        vertices("LN_var"), None, model.config, torch.zeros(len(state["LN_var"])))
    if stage == "qk":
        oa = upstream.OptimalQueryBiasVectors(sampler.key_names, model.config.n_embd // model.config.n_head, base)
        hooked = upstream.PruningModelWithHooksForQK(copy.deepcopy(model), config, sampler.mapping_to_param_idx, False, lambda *a: None)
        _qk_batch_compatibility(hooked)
    else:
        oa = base
        hooked = upstream.PruningModelWithHooksFullPaths(copy.deepcopy(model), config, sampler.mapping_to_param_idx, False, lambda *a: None)
    try:
        _load_state(oa, state)
        edges = {_tuple(row["edge"]): row["value"] for row in entry["masks"]}
        if (len(edges) != len(entry["masks"]) or set(edges) != set(sampler.mapping_to_param_idx)
                or any(type(v) is not int or v not in (0, 1) for v in edges.values())):
            raise ValueError("invalid stage masks")
        mask = torch.ones(len(edges))
        for edge, index in sampler.mapping_to_param_idx.items():
            mask[index] = edges[edge]
        return hooked(input_ids, masks=mask.expand(len(input_ids), -1), oa_vecs=oa, use_cache=False).logits
    finally:
        hooked.remove_hooks()
