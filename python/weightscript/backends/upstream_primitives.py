"""Weight-derived QK interaction matching against the actual upstream library.

Only direct embedding paths have a static matrix interpretation here. Other
learned paths remain opaque; numerical matrix fits are not equivalence proofs.
"""

import sys

import torch

from ..ir import Node
from .upstream import modules


@torch.no_grad()
def match_qk_primitives(model, artifact, checkout, *, tolerance=0.05, max_positions=64):
    if not 0 <= tolerance <= 1 or type(max_positions) is not int or max_positions < 1:
        raise ValueError("invalid primitive matching bounds")
    _, _, renderer = modules(checkout)
    data = artifact.to_dict()
    from ..core import fingerprint
    if data["provenance"]["checkpoint_sha256"] != fingerprint(model):
        raise ValueError("primitive evidence belongs to a different checkpoint")
    entry = data["nodes"][-1]["attributes"]["stages"][-1]
    if entry["name"] != "qk":
        raise ValueError("QK stage required for weight primitive matching")
    config = entry["config"]
    variances = {tuple(r["vertex"]): r["value"] for r in entry["auxiliary"]["LN_var"]}
    tokenizer = sys.modules["patching_data"].customTokenizer(
        [str(i) for i in range(model.config.vocab_size - 4)])
    embeddings = {"wte": model.transformer.wte.weight,
                  "wpe": model.transformer.wpe.weight[:max_positions]}
    nodes = []
    for layer in range(model.config.n_layer):
        block = model.transformer.h[layer]
        width = block.attn.head_dim
        for head in range(model.config.n_head):
            paths = config[str(layer)]["qk"][str(head)]
            bias_paths = config[str(layer)]["k"][str(head)]
            refs = [f"transformer.h.{layer}.attn.c_attn.weight",
                    f"transformer.h.{layer}.ln_1.weight"]
            if not paths and not bias_paths:
                nodes.append(Node(f"uniform.{layer}.{head}", "uniform_causal_selection", ["tokens"],
                                  {"layer": layer, "head": head, "evidence": "all QK and affine-key terms removed by upstream masks"},
                                  refs, "primitive"))
            for index, (q_name, k_name) in enumerate(paths):
                attrs = {"layer": layer, "head": head, "query_path": q_name, "key_path": k_name}
                matches = []
                if q_name in embeddings and k_name in embeddings:
                    def project(name, act, offset, block=block, layer=layer, head=head, width=width):
                        e = embeddings[name]
                        norm = (e - e.mean(-1, keepdim=True)) * block.ln_1.weight
                        norm = norm / (torch.tensor(variances[(layer, act, head)]).exp() + block.ln_1.eps).sqrt()
                        return norm @ block.attn.c_attn.weight[:, offset+head*width:offset+(head+1)*width]
                    q = project(q_name, "q", 0)
                    k = project(k_name, "k", model.config.n_embd)
                    actual = q @ k.T
                    # Softmax is invariant to a constant added across each row.
                    actual -= actual.mean(-1, keepdim=True)
                    actual_norm = float(actual.norm())
                    attrs.update({"shape": list(actual.shape), "tolerance": tolerance,
                                  "normalization": "calibrated linear LayerNorm; row constants quotiented out"})
                    if actual_norm < 1e-6:
                        attrs["claim"] = "row-centered QK contraction has no direction; left opaque"
                    else:
                        for primitive in renderer.ATTENTION_ALL_PRIMITIVES:
                            if primitive.is_only_token and (q_name != "wte" or k_name != "wte"):
                                continue
                            candidate = primitive.contruct_matrix(len(q), len(k), tokenizer).to(actual)
                            candidate -= candidate.mean(-1, keepdim=True)
                            denom = float(candidate.square().sum())
                            scale = float((actual * candidate).sum()) / denom if denom else 0.0
                            residual = float((actual - scale * candidate).norm())
                            operation = primitive.operation
                            matches.append({
                                "primitive": primitive.name,
                                "operation": operation() if callable(operation) else operation,
                                "scale": scale,
                                "relative_error": residual / actual_norm,
                            })
                        matches.sort(key=lambda m: (m["relative_error"], m["primitive"]))
                        attrs.update({"best_match": matches[0] if matches else None, "candidates": matches,
                                      "claim": "weight-contraction matrix fit; replacement fidelity not established"})
                else:
                    attrs["claim"] = "path is not a direct embedding contraction; left opaque"
                matched = bool(matches and matches[0]["relative_error"] <= tolerance)
                nodes.append(Node(f"qk.{layer}.{head}.{index}", "qk_weight_contraction", ["tokens"],
                                  attrs, refs + [f"transformer.{q_name}.weight", f"transformer.{k_name}.weight"]
                                  if q_name in embeddings and k_name in embeddings else refs, "hypothesis" if matched else "opaque"))
    artifact.nodes[-1].attributes["primitive_matches"] = [node.__dict__ for node in nodes]
    return nodes
