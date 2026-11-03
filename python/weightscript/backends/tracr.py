"""Safe JSON RASP frontend, real Tracr compiler and source-independent weight runtime."""

import copy
import hashlib
import itertools
import json
import math
import re
import random
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

from ..core import write_json

REVISION = "9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd"


def expression(expr, variables):
    if not isinstance(expr, dict):
        return expr
    if set(expr) == {"var"}:
        return variables[expr["var"]]
    op = expr.get("op")
    args = [expression(a, variables) for a in expr.get("args", [])]
    functions = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "floor_div": lambda a, b: a // b,
        "mod": lambda a, b: a % b,
        "max": max,
        "min": min,
        "eq": lambda a, b: a == b,
        "gt": lambda a, b: a > b,
        "lt": lambda a, b: a < b,
        "in": lambda a, b: a in b,
        "if": lambda c, a, b: a if c else b,
        "not": lambda a: not a,
        "and": lambda a, b: bool(a and b),
        "or": lambda a, b: bool(a or b),
    }
    if op not in functions:
        raise ValueError(f"unsupported RASP expression {op}")
    return functions[op](*args)


def build(spec):
    from tracr.rasp import rasp

    if spec.get("format") != "weightscript.rasp.v1":
        raise ValueError("unsupported RASP format")
    if (
        not spec.get("vocab")
        or not 1 <= spec["max_seq_len"] <= 256
        or not 1 <= len(spec["nodes"]) <= 256
    ):
        raise ValueError("invalid RASP domain")
    values = {}
    for node in spec["nodes"]:
        key, op = node["id"], node["op"]
        if key in values:
            raise ValueError("duplicate RASP node")
        refs = [values[name] for name in node.get("inputs", [])]
        if op == "tokens":
            result = rasp.tokens
        elif op == "indices":
            result = rasp.indices
        elif op == "map":
            result = rasp.Map(lambda x, e=node["expr"]: expression(e, {"x": x}), *refs)
        elif op == "sequence_map":
            result = rasp.SequenceMap(
                lambda x, y, e=node["expr"]: expression(e, {"x": x, "y": y}), *refs
            )
        elif op == "select":
            comparisons = {
                "eq": rasp.Comparison.EQ,
                "lt": rasp.Comparison.LT,
                "leq": rasp.Comparison.LEQ,
                "true": rasp.Comparison.TRUE,
            }
            result = rasp.Select(*refs, comparisons[node["comparison"]])
        elif op == "aggregate":
            result = rasp.Aggregate(*refs, default=node.get("default"))
        elif op == "selector_width":
            result = rasp.SelectorWidth(*refs)
        else:
            raise ValueError(f"unsupported RASP node {op}")
        if node.get("encoding") == "numerical":
            result = rasp.numerical(result)
        values[key] = result.named(key)
    return values[spec["output"]]


def compile_program(spec):
    from tracr.compiler import compiling
    from tracr.craft import bases
    from tracr.transformer.encoder import CategoricalEncoder

    program = build(spec)
    assembled = compiling.compile_rasp_to_model(
        program,
        vocab=set(spec["vocab"]),
        max_seq_len=spec["max_seq_len"],
        causal=spec.get("causal", True),
        compiler_bos="BOS",
        compiler_pad="PAD",
    )
    categorical = isinstance(assembled.output_encoder, CategoricalEncoder)
    if categorical:
        outputs = sorted(
            assembled.output_encoder.encoding_map,
            key=assembled.output_encoder.encoding_map.get,
        )
        columns = [
            assembled.residual_labels.index(str(bases.BasisDirection(program.label, v)))
            for v in outputs
        ]
    else:
        outputs = None
        columns = [
            assembled.residual_labels.index(str(bases.BasisDirection(program.label)))
        ]
    labels = [
        re.sub(r"_\d+$", "", label.split(":", 1)[0])
        + (":" + label.split(":", 1)[1] if ":" in label else "")
        for label in assembled.residual_labels
    ]
    config = assembled.model_config
    metadata = {
        "format": "weightscript.tracr.weights.v1",
        "compiler_revision": REVISION,
        "architecture": {
            k: getattr(config, k)
            for k in (
                "num_heads",
                "num_layers",
                "key_size",
                "mlp_hidden_size",
                "causal",
            )
        },
        "input_vocab": sorted(
            assembled.input_encoder.encoding_map,
            key=assembled.input_encoder.encoding_map.get,
        ),
        "output_vocab": outputs,
        "output_columns": columns,
        "residual_labels": labels,
        "max_seq_len": spec["max_seq_len"],
        "source": copy.deepcopy(spec),
    }
    params = {
        f"{module}::{name}": np.asarray(value, dtype=np.float32)
        for module, tensors in assembled.params.items()
        for name, value in tensors.items()
    }
    result = Compiled(metadata, params)
    # Fail closed on known upstream lowering limitations instead of assuming
    # every accepted RASP expression compiles faithfully.
    sanity = []
    for length in range(1, min(3, spec["max_seq_len"]) + 1):
        sanity.extend(
            itertools.islice(itertools.product(spec["vocab"], repeat=length), 256)
        )
    rng = random.Random(0)
    sanity.extend(
        tuple(rng.choice(spec["vocab"]) for _ in range(spec["max_seq_len"]))
        for _ in range(16)
    )
    for seq in sanity:
        actual = result.predict(list(seq))
        expected = program(list(seq))
        if spec.get("observation") == "last":
            actual = actual[-1:]
            expected = expected[-1:]
        same = (
            np.allclose(actual, expected, atol=1e-4, rtol=1e-4)
            if all(isinstance(x, (int, float)) for x in actual)
            else actual == expected
        )
        if not same:
            raise ValueError(
                f"Tracr lowering failed domain check on {list(seq)}: {actual} != {expected}"
            )
    metadata["validation"] = {
        "checked": len(sanity),
        "observation": spec.get("observation", "all"),
        "claim": "deterministic numerical smoke checks; not universal equivalence",
    }
    return result, assembled


class Compiled:
    def __init__(self, metadata, params):
        if metadata.get("format") != "weightscript.tracr.weights.v1":
            raise ValueError("unsupported compiled artifact")
        self.metadata, self.params = metadata, params

    def fingerprint(self):
        h = hashlib.sha256()
        # Deliberately exclude source and parameter-name provenance from identity.
        for field in (
            "architecture",
            "input_vocab",
            "output_vocab",
            "output_columns",
            "max_seq_len",
        ):
            h.update(json.dumps(self.metadata[field], sort_keys=True).encode())
        for name, w in sorted(self.params.items()):
            h.update(name.encode())
            h.update(str(w.shape).encode())
            h.update(w.tobytes())
        return h.hexdigest()

    def save(self, path):
        path = Path(path)
        if path.exists() and any(path.iterdir()):
            raise ValueError("output must be empty")
        path.mkdir(parents=True, exist_ok=True)
        save_file(self.params, str(path / "model.safetensors"))
        write_json(
            path / "manifest.json",
            dict(self.metadata, weights_sha256=self.fingerprint()),
        )

    @classmethod
    def load(cls, path):
        path = Path(path)
        result = cls(
            json.loads((path / "manifest.json").read_text()),
            load_file(str(path / "model.safetensors")),
        )
        if result.fingerprint() != result.metadata["weights_sha256"]:
            raise ValueError("compiled checkpoint hash mismatch")
        return result

    def predict_batch(self, sequences):
        # Independent inference from serialized weights, without source, JAX,
        # RASP rebuilding, or any call to the symbolic interpreter.
        if not sequences or not sequences[0] or len({len(x) for x in sequences}) != 1:
            raise ValueError("batch must contain equal-length nonempty sequences")
        if len(sequences[0]) > self.metadata["max_seq_len"]:
            raise ValueError("sequence exceeds compiled domain")
        vocab = self.metadata["input_vocab"]
        ids = np.array(
            [[vocab.index("BOS")] + [vocab.index(t) for t in seq] for seq in sequences]
        )
        w = self.params
        c = self.metadata["architecture"]
        x = (
            w["token_embed::embeddings"][ids]
            + w["pos_embed::embeddings"][np.arange(ids.shape[1])]
        )
        batch, length, _ = x.shape

        def linear(x, name):
            return x @ w[name + "::w"] + w[name + "::b"]

        mask = (
            np.tril(np.ones((length, length), dtype=bool))
            if c["causal"]
            else np.ones((length, length), dtype=bool)
        )
        for layer in range(c["num_layers"]):
            prefix = f"transformer/layer_{layer}"
            q, k, v = [
                linear(x, prefix + "/attn/" + name)
                .reshape(batch, length, c["num_heads"], c["key_size"])
                .transpose(0, 2, 1, 3)
                for name in ("query", "key", "value")
            ]
            scores = q @ k.swapaxes(-1, -2) / np.float32(math.sqrt(c["key_size"]))
            scores = np.where(mask, scores, np.float32(-1e30))
            probs = np.exp(scores - scores.max(-1, keepdims=True))
            probs /= probs.sum(-1, keepdims=True)
            joined = (probs @ v).transpose(0, 2, 1, 3).reshape(batch, length, -1)
            x = x + linear(joined, prefix + "/attn/linear")
            hidden = np.maximum(linear(x, prefix + "/mlp/linear_1"), 0)
            x = x + linear(hidden, prefix + "/mlp/linear_2")
        output = x[:, 1:, self.metadata["output_columns"]]
        if self.metadata["output_vocab"] is None:
            return output[..., 0].tolist()
        return [
            [self.metadata["output_vocab"][int(i)] for i in row]
            for row in output.argmax(-1)
        ]

    def predict(self, sequence):
        return self.predict_batch([sequence])[0]


def recover(model, library):
    """Restricted compiled-weight recognizer, using tensor equality, not stored source."""
    matches = []
    for spec in library:
        candidate, _ = compile_program(spec)
        if candidate.fingerprint() == model.fingerprint():
            matches.append(spec)
    if not matches:
        raise ValueError(
            "no exact program in the supplied library; refusing to invent one"
        )
    return {
        "program": matches[0],
        "matches": len(matches),
        "claim": "exact compiled tensor identity in supplied finite program library",
    }


def patch(original, spec):
    edited, _ = compile_program(spec)
    for field in (
        "architecture",
        "input_vocab",
        "output_vocab",
        "output_columns",
        "residual_labels",
        "max_seq_len",
    ):
        if original.metadata[field] != edited.metadata[field]:
            raise ValueError(f"edit changes {field}; use new-model compilation")
    if original.params.keys() != edited.params.keys() or any(
        original.params[k].shape != edited.params[k].shape for k in original.params
    ):
        raise ValueError("edit does not fit existing parameter layout")
    changed = {
        name: int(np.count_nonzero(original.params[name] != edited.params[name]))
        for name in original.params
    }
    return edited, {
        "architecture_preserved": True,
        "gradient_updates": 0,
        "changed_scalars": sum(changed.values()),
        "tensors": {name: n for name, n in changed.items() if n},
    }


def verify(model, spec, max_len=4):
    if (
        not 1 <= max_len <= spec["max_seq_len"]
        or sum(len(spec["vocab"]) ** n for n in range(1, max_len + 1)) > 100000
    ):
        raise ValueError("verification domain too large")
    program = build(spec)
    checked = 0
    for length in range(1, max_len + 1):
        sequences = list(itertools.product(spec["vocab"], repeat=length))
        for offset in range(0, len(sequences), 128):
            batch = sequences[offset : offset + 128]
            predicted = model.predict_batch(batch)
            for seq, actual in zip(batch, predicted):
                expected = program(list(seq))
                if spec.get("observation") == "last":
                    actual = actual[-1:]
                    expected = expected[-1:]
                checked += 1
                same = (
                    np.allclose(
                        np.array(actual), np.array(expected), atol=1e-4, rtol=1e-4
                    )
                    if all(isinstance(x, (int, float)) for x in actual)
                    else actual == expected
                )
                if not same:
                    return {
                        "passed": False,
                        "checked": checked,
                        "counterexample": list(seq),
                        "actual": actual,
                        "expected": expected,
                    }
    return {
        "passed": True,
        "checked": checked,
        "max_len": max_len,
        "claim": "exhaustive bounded output comparison, float32 inference; not a universal proof",
    }
