"""Versioned artifacts with explicit provenance, fidelity and opaque fallbacks."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Node:
    id: str
    op: str
    inputs: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    weights: list[str] = field(default_factory=list)
    interpretation: str = "numerical"


@dataclass
class Artifact:
    backend: str
    nodes: list[Node]
    output: str
    provenance: dict[str, Any]
    fidelity: dict[str, Any]
    version: str = "weightscript.ir.v1"

    def to_dict(self):
        self.validate()
        return asdict(self)

    def validate(self):
        seen = set()
        for node in self.nodes:
            if not node.id or node.id in seen:
                raise ValueError("duplicate or empty IR node")
            if any(parent not in seen for parent in node.inputs):
                raise ValueError("IR nodes must be in dependency order")
            if node.interpretation not in {
                "numerical",
                "hypothesis",
                "primitive",
                "opaque",
            }:
                raise ValueError("unknown interpretation status")
            seen.add(node.id)
        if self.output not in seen:
            raise ValueError("IR output is missing")
        if "claim" not in self.fidelity:
            raise ValueError("fidelity claim is required")


def graph_from_model(model):
    """Common graph for any HF module; uninterpreted blocks remain opaque."""
    from .core import fingerprint

    nodes = [Node("input", "token_ids")]
    # Preserve learned tensors by reference, never silently drop unknown modules.
    for name, module in model.named_modules():
        local = list(module.named_parameters(recurse=False))
        if local:
            nodes.append(
                Node(
                    name,
                    type(module).__name__,
                    ["input"],
                    {"shapes": {n: list(p.shape) for n, p in local}},
                    [f"{name}.{n}" if name else n for n, _ in local],
                    "opaque",
                )
            )
    nodes.append(
        Node(
            "output",
            "native_model",
            [n.id for n in nodes],
            {"model_type": model.config.model_type},
            interpretation="opaque",
        )
    )
    return Artifact(
        "native",
        nodes,
        "output",
        {"checkpoint_sha256": fingerprint(model)},
        {
            "claim": "module inventory; opaque execution retained, not an algorithm decompilation"
        },
    )
