"""Map direct checkpoint changes back to numerical program operations."""

import torch

from .core import audit, fingerprint
from .ir import Artifact, Node


def explain(before, after, max_elements=4_000_000):
    changes = audit(before, after)
    original = before.state_dict()
    edited = after.state_dict()
    nodes = [
        Node(
            "source",
            "checkpoint",
            attributes={"sha256": fingerprint(before)},
            interpretation="opaque",
        )
    ]
    for i, row in enumerate(changes["tensors"]):
        name = row["tensor"]
        a = original[name]
        b = edited[name]
        delta = b - a
        attrs = {"changed_scalars": row["changed"], "delta_l2": row["delta_l2"]}
        op = "tensor_update"
        status = "opaque"
        if a.ndim == 2 and a.numel() <= max_elements:
            active_rows = torch.where((delta != 0).any(1))[0]
            sub = delta[active_rows]
            _, singular, _ = torch.linalg.svd(sub, full_matrices=False)
            rank = (
                int((singular > max(float(singular[0]) * 1e-5, 1e-7)).sum())
                if singular.numel()
                else 0
            )
            attrs.update(
                active_rows=active_rows.tolist(),
                numerical_rank=rank,
                singular_values=singular[:16].tolist(),
                rank_relative_tolerance=1e-5,
            )
            if rank == 1:
                op = "rank_one_write_update"
                status = "numerical"
            denominator = (a * a).sum()
            if denominator > 0:
                factor = float((a * b).sum() / denominator)
                error = float((b - factor * a).abs().max())
                if torch.allclose(b, a * factor, atol=1e-6, rtol=1e-5):
                    op = "scale_projection"
                    status = "numerical"
                    attrs.update(factor=factor, max_reconstruction_error=error)
        nodes.append(Node(f"edit.{i}", op, ["source"], attrs, [name], status))
    nodes.append(
        Node(
            "output",
            "patched_checkpoint",
            [n.id for n in nodes],
            {"sha256": fingerprint(after)},
            interpretation="opaque",
        )
    )
    return Artifact(
        "patch.explanation",
        nodes,
        "output",
        {"audit": changes},
        {
            "claim": "numerical explanation of weight differences; semantic behavior is not inferred from low rank alone"
        },
    )
