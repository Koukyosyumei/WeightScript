"""Bounded local driver of the pinned upstream mask and MLP fitting loops.

The optimizer is the AdamW mask sampler, auxiliary-vector update, sparsity
penalty, and binary mask commit from ``patching.py``. MLP candidates are the
upstream multi-source primitives, traced and scored with the pinned functions.
HTCondor and ``delineate_curve_for_model`` are not imported. Step counts are
hard caps. A failed initial linear LayerNorm agreement rejects the stage even
when a later mask matches the first answer token.
"""

import copy
import importlib
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ..core import fingerprint
from ..evaluate import batches
from ..ir import Artifact, Node
from .upstream import (
    REVISION,
    URL,
    _engine,
    agreement_metrics,
    answer_logit_positions,
    empty_decision_counts,
    modules,
    score_decisions,
    stage_accepted,
)
from .upstream_stages import (
    _config,
    _load_state,
    _qk_batch_compatibility,
    _state,
    _tuple,
)


def _check_budget(
    steps,
    pretrain_steps,
    batch_size,
    max_seconds,
    min_fidelity,
    lr_sampler,
    lr_oa,
    lr_ln,
    sparsity,
    max_mlp_rows,
    max_feature_dim,
    max_behavior_tests,
):
    if (
        type(steps) is not int
        or not 0 <= steps <= 64
        or type(pretrain_steps) is not int
        or not 0 <= pretrain_steps <= 32
        or type(batch_size) is not int
        or not 1 <= batch_size <= 64
        or type(max_seconds) is not int
        or not 1 <= max_seconds <= 600
        or not 0 <= min_fidelity <= 1
        or not 0 < lr_sampler <= 1
        or not 0 < lr_oa <= 1
        or not 0 < lr_ln <= 1
        or not 0 <= sparsity <= 1
        or type(max_mlp_rows) is not int
        or not 1 <= max_mlp_rows <= 2048
        or type(max_feature_dim) is not int
        or not 1 <= max_feature_dim <= 2048
        or type(max_behavior_tests) is not int
        or not 0 <= max_behavior_tests <= 8
    ):
        raise ValueError("invalid coefficient-search budget")


def _check_rows(rows, name):
    if not rows:
        raise ValueError(f"{name} domain is empty")
    for row in rows:
        ids = row.get("input_ids")
        if (
            not isinstance(ids, list)
            or len(ids) < 2
            or any(type(token) is not int or token < 0 for token in ids)
        ):
            raise ValueError(f"invalid {name} input_ids")
        answer_logit_positions(len(ids), row.get("answer_start"))


def _load(checkout):
    pruning, helpers, renderer = modules(checkout)
    directory = Path(checkout).resolve() / "crasp/scripts/patching"
    sys.path.insert(0, str(directory))
    try:
        utils = importlib.import_module("patching_utils")
        convert_mlp = importlib.import_module("convert_mlp")
    finally:
        sys.path.remove(str(directory))
    return pruning, helpers, renderer, utils, convert_mlp


def _log_variances(model, sampler, groups):
    variances = {}
    hooks = []
    modules_by_key = {("final",): model.transformer.ln_f}
    for index, block in enumerate(model.transformer.h):
        modules_by_key[(index, "qkv")] = block.ln_1
        modules_by_key[(index, "mlp")] = block.ln_2
    for key, module in modules_by_key.items():
        variances[key] = []

        def remember(module, args, key=key):
            variances[key].append(args[0].var(-1, unbiased=False).mean().item())

        hooks.append(module.register_forward_pre_hook(remember))
    try:
        with torch.no_grad():
            for _, ids in groups:
                model(ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    values = torch.zeros(len(sampler.output_vertex))
    for index, vertex in enumerate(sampler.output_vertex):
        key = (
            ("final",)
            if vertex == ("lm_head",)
            else (vertex[0], "mlp" if vertex[1] == "mlp" else "qkv")
        )
        values[index] = (
            torch.tensor(sum(variances[key]) / len(variances[key]))
            .clamp_min(1e-12)
            .log()
        )
    return values


def _targets(model, groups):
    with torch.no_grad():
        return [model(ids, use_cache=False).logits.detach() for _, ids in groups]


def _measure(hooked, oa, mask, groups, targets):
    counts = empty_decision_counts()
    error = 0.0
    with torch.no_grad():
        for (rows, ids), target in zip(groups, targets):
            actual = hooked(
                ids, masks=mask.expand(len(ids), -1), oa_vecs=oa, use_cache=False
            ).logits
            error = max(error, float((actual - target).abs().max()))
            score_decisions(rows, actual, target, counts)
    return agreement_metrics(counts, error)


def _answer_kl(actual, target, rows):
    pieces = []
    for index, row in enumerate(rows):
        positions = answer_logit_positions(
            len(row["input_ids"]), row.get("answer_start")
        )
        pred = actual[index, positions]
        gold = target[index, positions]
        pieces.append(
            F.kl_div(
                pred.log_softmax(-1),
                gold.log_softmax(-1),
                log_target=True,
                reduction="none",
            ).sum(-1)
        )
    return torch.cat(pieces).mean()


def _ordinary_parameters(module):
    """Replace inference-mode parameters so the auxiliary AdamW step can run.

    ``_engine`` builds the component ablation vectors inside inference mode.
    Those tensors cannot re-enable gradients. The values are unchanged.
    """
    for child in module.modules():
        for key, parameter in list(child._parameters.items()):
            if parameter is not None:
                child._parameters[key] = torch.nn.Parameter(parameter.detach().clone())


def _snapshot(module):
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }


def _restore(module, saved):
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            parameter.copy_(saved[name])


def _param_summary(sampler):
    values = sampler.sample_params.detach()
    if values.numel() == 0:
        return {"edges": 0}
    return {
        "edges": int(values.numel()),
        "min": float(values.min()),
        "max": float(values.max()),
        "fraction_above_zero": float((values > 0).float().mean()),
        "uncertain_unit_interval": int(((values > -1) & (values < 1)).sum()),
    }


def _fit_masks(
    hooked,
    sampler,
    oa,
    model,
    groups,
    *,
    stage,
    steps,
    pretrain_steps,
    lr_sampler,
    lr_oa,
    lr_ln,
    sparsity,
    seed,
    deadline,
):
    """Run the upstream AdamW loop and return a binary mask plus a meter.

    The paper resets a countdown while any mask logit stays in (-1, 1) and can
    also stop on a caller-supplied baseline loss. This driver does not extend
    past ``steps`` and does not read an external baseline. Pretraining is the
    same auxiliary objective, capped by ``pretrain_steps`` rather than 500.
    """
    torch.manual_seed(seed + stage)
    meter = {
        "stage": stage,
        "optimizer_steps": 0,
        "pretrain_steps": 0,
        "hooked_forwards": 0,
        "native_forwards": 0,
        "stop": None,
        "losses": [],
    }
    if sampler.sample_params.numel() == 0:
        meter["stop"] = "no_edges"
        meter["parameters"] = _param_summary(sampler)
        return torch.ones(0), meter

    def expired():
        return time.monotonic() > deadline

    def native_batch(ids):
        meter["native_forwards"] += 1
        with torch.no_grad():
            return model(ids, use_cache=False).logits.detach()

    if stage > 0 and pretrain_steps:
        oa.requires_grad_(False)
        trainable = [oa.output_vertex_oa]
        if stage == 2 and oa.q_bias_term.numel():
            trainable.append(oa.q_bias_term)
        for parameter in trainable:
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(trainable, lr=lr_oa, weight_decay=0)
        for _ in range(pretrain_steps):
            if expired():
                meter["stop"] = "time_limit"
                break
            rows, ids = groups[meter["pretrain_steps"] % len(groups)]
            target = native_batch(ids)
            optimizer.zero_grad(set_to_none=True)
            masks = sampler.sample_binary_masks(len(ids))
            actual = hooked(ids, masks=masks, oa_vecs=oa, use_cache=False).logits
            meter["hooked_forwards"] += 1
            loss = _answer_kl(actual, target, rows)
            if not torch.isfinite(loss):
                meter["stop"] = "nonfinite_pretrain_loss"
                break
            loss.backward()
            optimizer.step()
            meter["pretrain_steps"] += 1
            meter["losses"].append(float(loss.detach()))
        for parameter in oa.parameters():
            parameter.grad = None

    oa.requires_grad_(False)
    if meter["stop"] is None and steps:
        if stage <= 1:
            groups_oa = [
                {"params": [oa.LN_var], "lr": lr_ln},
                {"params": [oa.input_vertex_oa], "lr": lr_oa},
                {"params": [oa.output_vertex_oa], "lr": 1e-4},
            ]
        else:
            groups_oa = [
                {"params": [oa.LN_var], "lr": lr_ln},
                {"params": [oa.output_vertex_oa], "lr": 1e-4},
            ]
            if oa.q_bias_term.numel():
                groups_oa.append({"params": [oa.q_bias_term], "lr": lr_oa})
        for group in groups_oa:
            for parameter in group["params"]:
                parameter.requires_grad_(True)
        sampler_optimizer = torch.optim.AdamW(
            sampler.parameters(), lr=lr_sampler, weight_decay=0, betas=(0.9, 0.995)
        )
        oa_optimizer = torch.optim.AdamW(groups_oa, weight_decay=0)
        for _ in range(steps):
            if expired():
                meter["stop"] = "time_limit"
                break
            rows, ids = groups[meter["optimizer_steps"] % len(groups)]
            saved_sampler = _snapshot(sampler)
            saved_oa = _snapshot(oa)
            target = native_batch(ids)
            sampler_optimizer.zero_grad(set_to_none=True)
            oa_optimizer.zero_grad(set_to_none=True)
            masks = sampler.sample_masks(len(ids))
            actual = hooked(ids, masks=masks, oa_vecs=oa, use_cache=False).logits
            meter["hooked_forwards"] += 1
            loss = _answer_kl(actual, target, rows)
            penalty, _parts = sampler.get_penalty(0.0)
            loss = loss + sparsity * penalty
            if not torch.isfinite(loss):
                _restore(sampler, saved_sampler)
                _restore(oa, saved_oa)
                meter["stop"] = "nonfinite_loss"
                break
            try:
                loss.backward()
            except RuntimeError as exc:
                _restore(sampler, saved_sampler)
                _restore(oa, saved_oa)
                meter["stop"] = f"backward_failed: {exc}"
                break
            torch.nn.utils.clip_grad_norm_(sampler.parameters(), 5)
            sampler_optimizer.step()
            oa_optimizer.step()
            with torch.no_grad():
                if oa.LN_var.abs().max() > 80:
                    oa.LN_var.clamp_(-80, 80)
                    meter["ln_clamped"] = True
            meter["optimizer_steps"] += 1
            meter["losses"].append(float(loss.detach()))
            if not torch.isfinite(sampler.sample_params).all():
                _restore(sampler, saved_sampler)
                _restore(oa, saved_oa)
                meter["stop"] = "nonfinite_mask_logits"
                break
            if float(sampler.sample_params.detach().max()) < -2:
                meter["stop"] = "all_mask_logits_below_-2"
                break
    oa.requires_grad_(False)
    with torch.no_grad():
        binary = sampler.sample_binary_masks(1).squeeze(0).detach()
    if meter["stop"] is None:
        meter["stop"] = "step_cap"
    meter["parameters"] = _param_summary(sampler)
    meter["binary_edges"] = int(binary.sum()) if binary.numel() else 0
    return binary, meter


def _blocked_mlp(name, converted):
    for match in re.findall(r"mlp-\d+", name):
        if match not in converted:
            return True
    return False


def _fvu(predicted, actual):
    error = (predicted - actual).pow(2).mean()
    variance = actual.var(dim=0).mean()
    if float(variance) < 1e-12:
        return 0.0 if float(error) < 1e-12 else float("inf")
    return float(error / variance)


def _behavior(hooked, oa, mask, groups, targets, converted):
    hooked.set_converted_mlp(converted)
    try:
        return _measure(hooked, oa, mask, groups, targets)
    finally:
        if hasattr(hooked, "converted_mlp"):
            hooked.remove_converted_mlp()


def _replace_mlps(
    hooked,
    oa,
    mask,
    groups,
    targets,
    ood_groups,
    ood_targets,
    convert_mlp,
    primitive_fn,
    *,
    min_fidelity,
    max_mlp_rows,
    max_feature_dim,
    max_behavior_tests,
    deadline,
):
    """Fit upstream MLP primitives and keep one only after causal and held-out checks.

    Calibration rows choose ``C``. The disjoint rows never enter ``pinv``. A
    candidate stays opaque unless replacing the MLP inside the QK engine matches
    both the unreplaced engine and the frozen model on every requested answer
    position in both domains.
    """
    report = {"accepted": [], "rejected": [], "probe_forwards": 0, "selector": []}
    if not ood_groups:
        report["rejected"].append({"reason": "no disjoint out-of-domain rows"})
        return {}, report
    converted = {}
    for layer in range(len(hooked.model.transformer.h)):
        if time.monotonic() > deadline:
            report["rejected"].append({"layer": layer, "reason": "time_limit"})
            break
        inputs = list(hooked.config[layer]["mlp"])
        path = f"mlp-{layer}"
        if not inputs:
            report["rejected"].append(
                {"path": path, "reason": "no mlp inputs in the committed mask"}
            )
            continue
        if any(_blocked_mlp(name, converted) for name in inputs):
            report["rejected"].append(
                {"path": path, "reason": "dependency on an unconverted mlp"}
            )
            continue
        collected_in = []
        collected_out = []
        selector = "answer_positions"
        trace_failed = False
        for rows, ids in groups:
            if sum(item.shape[0] for item in collected_out) >= max_mlp_rows:
                break
            if time.monotonic() > deadline:
                break
            position_ids = torch.arange(ids.shape[1]).unsqueeze(0).expand(len(ids), -1)
            try:
                with torch.no_grad():
                    hooked(
                        ids,
                        position_ids=position_ids,
                        masks=mask.expand(len(ids), -1),
                        oa_vecs=oa,
                        use_cache=False,
                    )
                    report["probe_forwards"] += 1
                    traced = convert_mlp.trace_mlp_multi_source(
                        hooked, converted, path, ids, position_ids, lambda *_args: None
                    )
                    mlp_out = hooked.activations[path].detach()
            except (RuntimeError, NotImplementedError, KeyError, IndexError) as exc:
                report["rejected"].append(
                    {"path": path, "reason": f"trace failed: {exc}"}
                )
                trace_failed = True
                break
            if any(not torch.is_tensor(item) for item in traced):
                raise RuntimeError(
                    f"upstream MLP trace returned a non-tensor for {path}"
                )
            chosen = []
            for row_index, row in enumerate(rows):
                for position in answer_logit_positions(
                    len(row["input_ids"]), row.get("answer_start")
                ):
                    chosen.append((row_index, position))
                    if (
                        sum(item.shape[0] for item in collected_out) + len(chosen)
                        >= max_mlp_rows
                    ):
                        break
            if not chosen:
                continue
            index = torch.tensor(chosen)
            collected_in.append(
                [item[index[:, 0], index[:, 1]].detach() for item in traced]
            )
            collected_out.append(mlp_out[index[:, 0], index[:, 1]])
            selector = "answer_positions"
        report["selector"].append({"path": path, "selector": selector})
        if trace_failed:
            continue
        if not collected_out:
            report["rejected"].append({"path": path, "reason": "no probe rows"})
            continue
        features = [torch.cat(items, dim=0) for items in zip(*collected_in)]
        observed = torch.cat(collected_out, dim=0)
        if not all(
            torch.allclose(item.sum(-1), torch.ones(item.shape[0]), atol=1e-2)
            for item in features
        ):
            report["rejected"].append(
                {
                    "path": path,
                    "reason": "traced MLP inputs are not one-hot distributions",
                    "rows": int(observed.shape[0]),
                }
            )
            continue
        candidates = ["erase"] + [("keep_one", index) for index in range(len(features))]
        if len(features) > 1:
            candidates.append("combine")
        scored = []
        for primitive in candidates:
            try:
                transformed = primitive_fn(features, primitive=primitive)
            except NotImplementedError as exc:
                scored.append({"primitive": primitive, "reason": str(exc)})
                continue
            width = int(transformed.shape[-1])
            if (
                width > max_feature_dim
                or transformed.shape[0] < 2
                or not torch.isfinite(transformed).all()
            ):
                scored.append(
                    {"primitive": primitive, "reason": "feature bound", "width": width}
                )
                continue
            try:
                coefficient = torch.linalg.pinv(transformed) @ observed
            except RuntimeError as exc:
                scored.append({"primitive": primitive, "reason": f"pinv failed: {exc}"})
                continue
            if not torch.isfinite(coefficient).all():
                scored.append(
                    {"primitive": primitive, "reason": "nonfinite coefficient"}
                )
                continue
            residual = _fvu(transformed @ coefficient, observed)
            scored.append(
                {
                    "primitive": primitive,
                    "width": width,
                    "rows": int(transformed.shape[0]),
                    "fvu": residual,
                    "coefficient": coefficient.detach(),
                }
            )
        fitted = [item for item in scored if "coefficient" in item]
        fitted.sort(key=lambda item: (item["fvu"], str(item["primitive"])))
        viable = [item for item in fitted if item["fvu"] < 0.6]
        best = None
        tests = 0
        for item in viable:
            if tests >= max_behavior_tests or time.monotonic() > deadline:
                break
            tests += 1
            trial = dict(converted)
            trial[path] = (item["primitive"], item["coefficient"])
            try:
                calibration = _behavior(hooked, oa, mask, groups, targets, trial)
                heldout = _behavior(hooked, oa, mask, ood_groups, ood_targets, trial)
                item["causal_agreement"] = _causal_agreement(
                    hooked, oa, mask, groups, trial, dict(converted)
                )
            except (RuntimeError, ValueError, KeyError, IndexError) as exc:
                item["behavior_error"] = str(exc)
                continue
            item["calibration_vs_native"] = calibration["answer_position_agreement"]
            item["heldout_vs_native"] = heldout["answer_position_agreement"]
            item["calibration_sequence"] = calibration["answer_sequence_agreement"]
            item["heldout_sequence"] = heldout["answer_sequence_agreement"]
            passed = (
                item["calibration_vs_native"] >= min_fidelity
                and item["heldout_vs_native"] >= min_fidelity
                and item["causal_agreement"] >= min_fidelity
            )
            item["passed"] = passed
            if passed and (best is None or item["fvu"] < best["fvu"]):
                best = item
        if best is None:
            fallback = (
                fitted[0]
                if fitted
                else (scored[0] if scored else {"reason": "no primitive"})
            )
            rejection = _public_fit(path, fallback)
            if fitted and fitted[0]["fvu"] >= 0.6:
                rejection["reason"] = (
                    "best upstream primitive FVU is at least 0.6; left opaque"
                )
            rejection["behavior_tests"] = tests
            report["rejected"].append(rejection)
            continue
        converted[path] = (best["primitive"], best["coefficient"])
        report["accepted"].append(_public_fit(path, best, keep_coefficient=True))
    return converted, report


def _run_converted(hooked, oa, mask, ids, converted):
    if converted:
        hooked.set_converted_mlp(converted)
    try:
        return hooked(
            ids, masks=mask.expand(len(ids), -1), oa_vecs=oa, use_cache=False
        ).logits
    finally:
        if hasattr(hooked, "converted_mlp"):
            hooked.remove_converted_mlp()


def _causal_agreement(hooked, oa, mask, groups, trial, baseline):
    """Answer-position agreement between the replaced MLP and the unreplaced engine."""
    counts = empty_decision_counts()
    with torch.no_grad():
        for rows, ids in groups:
            before = _run_converted(hooked, oa, mask, ids, baseline)
            after = _run_converted(hooked, oa, mask, ids, trial)
            score_decisions(rows, after, before, counts)
    return agreement_metrics(counts)["answer_position_agreement"]


def _public_fit(path, item, keep_coefficient=False):
    primitive = item.get("primitive")
    public = {
        "path": path,
        "primitive": list(primitive) if isinstance(primitive, tuple) else primitive,
        "fvu": item.get("fvu"),
        "width": item.get("width"),
        "rows": item.get("rows"),
        "reason": item.get("reason"),
        "calibration_vs_native": item.get("calibration_vs_native"),
        "heldout_vs_native": item.get("heldout_vs_native"),
        "causal_agreement": item.get("causal_agreement"),
        "passed": bool(item.get("passed", False)),
    }
    if keep_coefficient and "coefficient" in item:
        public["coefficient"] = item["coefficient"].detach().cpu().tolist()
    return {key: value for key, value in public.items() if value is not None}


def _render(upstream, renderer, name, config, heads):
    serial = json_config(config)
    if name == "component":
        return renderer.convert_config_to_code_one_step_paths(serial), None
    if name == "qk":
        return renderer.convert_config_to_code_qk_pruned(serial, False, None), None
    try:
        return renderer.convert_config_to_code_full_paths(serial), None
    except UnboundLocalError as exc:
        if "labels" not in str(exc):
            raise
        display = copy.deepcopy(config)
        upstream.convert_full_paths_config_to_prune_inside_kq_config_(display, heads)
        source = renderer.convert_config_to_code_qk_pruned(
            json_config(display), False, None
        )
        return (
            source,
            "upstream full-path renderer has undefined heatmap labels; rendered equivalent expanded config with upstream QK renderer",
        )


def json_config(config):
    import json

    return json.loads(json.dumps(config))


def _masks(mask, mapping):
    return [
        {"edge": edge, "value": int(mask[index])} for edge, index in mapping.items()
    ]


def decompile_coefficient(
    model,
    checkout,
    rows,
    evaluation_rows=None,
    *,
    steps=8,
    pretrain_steps=2,
    batch_size=8,
    learning_rate_sampler=0.1,
    learning_rate_oa=0.002,
    learning_rate_ln=0.1,
    sparsity=1e-3,
    min_fidelity=0.99,
    seed=0,
    max_seconds=180,
    max_mlp_rows=256,
    max_feature_dim=512,
    max_behavior_tests=2,
):
    """Search masks with the pinned sampler and then test MLP primitive replacement.

    Source weights stay frozen. Oracle rows are scored after the binary mask is
    committed and are the only out-of-domain set for an MLP replacement. The
    returned stage is rejected when the initial linear LayerNorm engine misses
    the requested answer span.
    """
    _check_budget(
        steps,
        pretrain_steps,
        batch_size,
        max_seconds,
        min_fidelity,
        learning_rate_sampler,
        learning_rate_oa,
        learning_rate_ln,
        sparsity,
        max_mlp_rows,
        max_feature_dim,
        max_behavior_tests,
    )
    if (
        model.config.model_type != "gpt2"
        or model.training
        or next(model.parameters()).device.type != "cpu"
    ):
        raise ValueError("coefficient search requires an eval-mode CPU GPT-2")
    _check_rows(rows, "calibration")
    if evaluation_rows:
        _check_rows(evaluation_rows, "evaluation")
        if {tuple(row["input_ids"]) for row in rows} & {
            tuple(row["input_ids"]) for row in evaluation_rows
        }:
            raise ValueError("calibration and evaluation domains overlap")
    upstream, helpers, renderer, utils, convert_mlp = _load(checkout)
    primitive_fn = utils.get_mlp_primitives_multi_source
    source_hash = fingerprint(model)
    deadline = time.monotonic() + max_seconds
    groups = list(batches(rows, batch_size))
    tests = list(batches(evaluation_rows, batch_size)) if evaluation_rows else []
    targets = _targets(model, groups)
    test_targets = _targets(model, tests) if tests else []
    heads = {index: model.config.n_head for index in range(model.config.n_layer)}
    config = helpers.get_full_possible_config_for_pruning(heads)
    copy_model = copy.deepcopy(model).eval()
    sampler = upstream.MaskSampler(config)
    variances = _log_variances(copy_model, sampler, groups)
    hooked, oa = _engine(model, upstream, config, sampler, variances, True, copy_model)
    _ordinary_parameters(oa)
    stages = []
    meters = []
    mlp_report = None
    previous_oa = None
    try:
        for name, stage_index in (("component", 0), ("full_path", 1), ("qk", 2)):
            if name == "full_path":
                upstream.convert_config_fullpaths_(config, heads)
                sampler = upstream.MaskSamplerFullPaths(config, False)
                values = []
                for vertex in sampler.all_output_vertex:
                    original = vertex[:3] if len(vertex) == 4 else vertex
                    values.append(
                        previous_oa.LN_var[previous_oa.to_LN_idx[original]].detach()
                    )
                oa = upstream.OptimalAblationVectors(
                    sampler.input_vertex,
                    sampler.output_vertex,
                    sampler.all_output_vertex,
                    None,
                    model.config,
                    torch.stack(values),
                )
                with torch.no_grad():
                    for vertex in oa.output_vertex:
                        if vertex in previous_oa.to_out_oa_idx:
                            oa.output_vertex_oa[oa.to_out_oa_idx[vertex]].copy_(
                                previous_oa.output_vertex_oa[
                                    previous_oa.to_out_oa_idx[vertex]
                                ]
                            )
                hooked.remove_hooks()
                hooked = upstream.PruningModelWithHooksFullPaths(
                    copy.deepcopy(model),
                    config,
                    sampler.mapping_to_param_idx,
                    False,
                    lambda *_args: None,
                )
            elif name == "qk":
                upstream.convert_full_paths_config_to_prune_inside_kq_config_(
                    config, heads
                )
                sampler = upstream.MaskSamplerForQK(config)
                oa = upstream.OptimalQueryBiasVectors(
                    sampler.key_names,
                    model.config.n_embd // model.config.n_head,
                    previous_oa,
                )
                with torch.no_grad():
                    for layer, head, key in sampler.key_names:
                        attention = model.transformer.h[layer].attn
                        width = attention.head_dim
                        start = head * width
                        vector = previous_oa.output_vertex_oa[
                            previous_oa.to_out_oa_idx[(layer, "q", head)]
                        ]
                        norm = model.transformer.h[layer].ln_1
                        variance = previous_oa.LN_var[
                            previous_oa.to_LN_idx[(layer, "q", head)]
                        ].exp()
                        normalized = (
                            (vector - vector.mean())
                            / (variance + norm.eps).sqrt()
                            * norm.weight
                        )
                        oa.q_bias_term[oa.to_q_bias[(layer, head, key)]].copy_(
                            normalized
                            @ attention.c_attn.weight[:, start : start + width]
                            + attention.c_attn.bias[start : start + width]
                        )
                hooked.remove_hooks()
                hooked = upstream.PruningModelWithHooksForQK(
                    copy.deepcopy(model),
                    config,
                    sampler.mapping_to_param_idx,
                    False,
                    lambda *_args: None,
                )
                _qk_batch_compatibility(hooked)
            engine_config = copy.deepcopy(config)
            initial = _measure(
                hooked,
                oa,
                torch.ones(len(sampler.mapping_to_param_idx)),
                groups,
                targets,
            )
            binary, meter = _fit_masks(
                hooked,
                sampler,
                oa,
                model,
                groups,
                stage=stage_index,
                steps=steps,
                pretrain_steps=pretrain_steps if stage_index else 0,
                lr_sampler=learning_rate_sampler,
                lr_oa=learning_rate_oa,
                lr_ln=learning_rate_ln,
                sparsity=sparsity,
                seed=seed,
                deadline=deadline,
            )
            meters.append(meter)
            if binary.numel() != len(sampler.mapping_to_param_idx):
                raise RuntimeError("binary mask does not match the upstream sampler")
            final = _measure(hooked, oa, binary, groups, targets)
            validation = (
                _measure(hooked, oa, binary, tests, test_targets) if tests else None
            )
            decision = stage_accepted(initial, final, validation, min_fidelity)
            if name == "qk":
                converter = upstream.convert_mask_to_config_qk_
            else:
                converter = upstream.convert_mask_to_config_
            converter(binary, config, sampler.mapping_to_param_idx)
            source, warning = _render(upstream, renderer, name, config, heads)
            stage = {
                "name": name,
                "engine_config": engine_config,
                "config": json_config(config),
                "auxiliary": _state(oa),
                "source": source,
                "renderer_adaptation": warning,
                "masks": _masks(binary, sampler.mapping_to_param_idx),
                "initial": initial,
                "calibration": final,
                "evaluation": validation,
                "meter": meter,
                "edges_before": int(binary.numel()),
                "edges_after": int(binary.sum()),
                "search_complete": meter["stop"] != "time_limit",
                "normalization_assumption_passed": decision[
                    "normalization_assumption_passed"
                ],
                "accepted": decision["accepted"] and meter["stop"] != "time_limit",
            }
            if name == "qk":
                _converted, mlp_report = _replace_mlps(
                    hooked,
                    oa,
                    binary,
                    groups,
                    targets,
                    tests,
                    test_targets,
                    convert_mlp,
                    primitive_fn,
                    min_fidelity=min_fidelity,
                    max_mlp_rows=max_mlp_rows,
                    max_feature_dim=max_feature_dim,
                    max_behavior_tests=max_behavior_tests,
                    deadline=deadline,
                )
                stage["mlp_replacement"] = {
                    "accepted": mlp_report["accepted"],
                    "rejected": mlp_report["rejected"],
                    "probe_forwards": mlp_report["probe_forwards"],
                    "selector": mlp_report["selector"],
                    "installed": [row["path"] for row in mlp_report["accepted"]],
                }
            stages.append(stage)
            previous_oa = oa
            if meter["stop"] == "time_limit":
                break
    finally:
        hooked.remove_hooks()
    if fingerprint(model) != source_hash:
        raise RuntimeError("coefficient search mutated frozen source weights")
    adaptations = ["broadcast cached positional embeddings across QK batches"]
    if any(stage.get("renderer_adaptation") for stage in stages):
        adaptations.append(
            "QK renderer fallback for full-path renderer undefined heatmap labels"
        )
    adaptations.append("full-path output ablation warm-started from the previous stage")
    adaptations.append(
        "QK affine query bias warm-started from the full-path query vector, then AdamW"
    )
    artifact = Artifact(
        "decompiling_transformers.coefficient_search",
        [
            Node("tokens", "tokens", interpretation="primitive"),
            Node(
                "output",
                "upstream_coefficient_search",
                ["tokens"],
                {"stages": stages},
                [name for name, _parameter in model.named_parameters()],
                "opaque",
            ),
        ],
        "output",
        {
            "repository": URL,
            "revision": REVISION,
            "checkpoint_sha256": source_hash,
            "driver": "bounded AdamW MaskSampler and auxiliary update from patching.py; MLP primitives from convert_mlp.py; HTCondor not used",
            "completed_stages": [stage["name"] for stage in stages],
            "source_weights_frozen": True,
            "split_mlps": False,
            "compatibility_adaptations": adaptations,
            "limits": {
                "steps": steps,
                "pretrain_steps": pretrain_steps,
                "batch_size": batch_size,
                "max_seconds": max_seconds,
                "sparsity": sparsity,
                "seed": seed,
                "countdown_extension": False,
                "baseline_loss_stop": False,
                "paper_pretrain_steps": 500,
            },
        },
        {
            "claim": "measured answer-position fidelity of a capped local coefficient search; not a paper reproduction or logit equivalence",
            "normalization": {
                "positions": "implicit positions 0..length-1",
                "layernorm": "per-vertex calibration log-variance, then AdamW on LN_var within the step cap",
                "acceptance": "every requested answer position, and the initial linear LayerNorm engine must already match",
            },
            "accepted": bool(stages)
            and all(stage["accepted"] for stage in stages)
            and len(stages) == 3,
            "meters": meters,
            "mlp_replacement": None
            if mlp_report is None
            else {
                "installed": [row["path"] for row in mlp_report["accepted"]],
                "rejected": len(mlp_report["rejected"]),
                "probe_forwards": mlp_report["probe_forwards"],
            },
        },
    )
    return artifact


def _restore_primitive(value):
    if isinstance(value, list):
        return tuple(value)
    return value


@torch.no_grad()
def execute_coefficient(
    model, artifact, checkout, input_ids, stage="qk", apply_mlp=True
):
    """Replay one coefficient-search stage, including an accepted MLP replacement."""
    data = artifact.to_dict() if isinstance(artifact, Artifact) else artifact
    if (
        data.get("backend") != "decompiling_transformers.coefficient_search"
        or data.get("version") != "weightscript.ir.v1"
        or data.get("provenance", {}).get("checkpoint_sha256") != fingerprint(model)
    ):
        raise ValueError("unsupported or checkpoint-mismatched coefficient artifact")
    if stage not in ("component", "full_path", "qk"):
        raise ValueError("replay stage must be component, full_path, or qk")
    entry = next(
        item
        for item in data["nodes"][-1]["attributes"]["stages"]
        if item["name"] == stage
    )
    upstream, _, _ = modules(checkout)
    config = _config(entry["engine_config"])
    if stage == "component":
        sampler = upstream.MaskSampler(config)
        hooked_oa = upstream.OptimalAblationVectors(
            [_tuple(row["vertex"]) for row in entry["auxiliary"]["input_vertex_oa"]],
            [_tuple(row["vertex"]) for row in entry["auxiliary"]["output_vertex_oa"]],
            [_tuple(row["vertex"]) for row in entry["auxiliary"]["LN_var"]],
            None,
            model.config,
            torch.zeros(len(entry["auxiliary"]["LN_var"])),
        )
        hooked = upstream.PruningModelWithHooks(
            copy.deepcopy(model),
            config,
            sampler.mapping_to_param_idx,
            lambda *_args: None,
            True,
        )
        oa = hooked_oa
    elif stage == "full_path":
        sampler = upstream.MaskSamplerFullPaths(config, False)
        oa = upstream.OptimalAblationVectors(
            [_tuple(row["vertex"]) for row in entry["auxiliary"]["input_vertex_oa"]],
            [_tuple(row["vertex"]) for row in entry["auxiliary"]["output_vertex_oa"]],
            [_tuple(row["vertex"]) for row in entry["auxiliary"]["LN_var"]],
            None,
            model.config,
            torch.zeros(len(entry["auxiliary"]["LN_var"])),
        )
        hooked = upstream.PruningModelWithHooksFullPaths(
            copy.deepcopy(model),
            config,
            sampler.mapping_to_param_idx,
            False,
            lambda *_args: None,
        )
    else:
        for layer in range(model.config.n_layer):
            for head in config[layer]["qk"]:
                config[layer]["qk"][head] = [
                    _tuple(pair) for pair in config[layer]["qk"][head]
                ]
        sampler = upstream.MaskSamplerForQK(config)
        base = upstream.OptimalAblationVectors(
            [],
            [_tuple(row["vertex"]) for row in entry["auxiliary"]["output_vertex_oa"]],
            [_tuple(row["vertex"]) for row in entry["auxiliary"]["LN_var"]],
            None,
            model.config,
            torch.zeros(len(entry["auxiliary"]["LN_var"])),
        )
        oa = upstream.OptimalQueryBiasVectors(
            sampler.key_names, model.config.n_embd // model.config.n_head, base
        )
        hooked = upstream.PruningModelWithHooksForQK(
            copy.deepcopy(model),
            config,
            sampler.mapping_to_param_idx,
            False,
            lambda *_args: None,
        )
        _qk_batch_compatibility(hooked)
    try:
        _load_state(oa, entry["auxiliary"])
        edges = {_tuple(row["edge"]): row["value"] for row in entry["masks"]}
        if set(edges) != set(sampler.mapping_to_param_idx) or any(
            value not in (0, 1) for value in edges.values()
        ):
            raise ValueError("invalid coefficient masks")
        mask = torch.ones(len(edges))
        for edge, index in sampler.mapping_to_param_idx.items():
            mask[index] = edges[edge]
        replacement = entry.get("mlp_replacement") or {}
        if apply_mlp and stage == "qk" and replacement.get("accepted"):
            installed = {}
            for row in replacement["accepted"]:
                installed[row["path"]] = (
                    _restore_primitive(row["primitive"]),
                    torch.tensor(row["coefficient"], dtype=torch.float32),
                )
            hooked.set_converted_mlp(installed)
        return hooked(
            input_ids,
            masks=mask.expand(len(input_ids), -1),
            oa_vecs=oa,
            use_cache=False,
        ).logits
    finally:
        hooked.remove_hooks()
