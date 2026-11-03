"""Bounded, resumable MATH-500 / GSM8K runs and untouched-versus-edited comparison.

The generator is a callable the caller supplies. This module does not import a
model, vLLM, or an API client. Gold answers are not placed on generation requests.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .data import ExampleSet, SplitIsolationError
from .extract import grade_response, math_verify_status
from .protocol import (
    FORMAT_VERSION,
    QWEN35_BASE_MODEL,
    QWEN35_BASE_REVISION,
    ProtocolError,
    protocol,
    user_turn,
)


class ConfigMismatch(ValueError):
    """The results file is bound to a different model, dataset, or protocol."""


class IntegrityError(ValueError):
    """The results file contradicts itself or the dataset it claims."""


class GenerationRejected(RuntimeError):
    """A generation did not finish with stop or length, so it was not stored."""


_ARM_FIELDS = ("arm", "checkpoint_id", "checkpoint_fingerprint")
_ACCEPTED_FINISH = ("stop", "length")


@dataclass(frozen=True)
class Budgets:
    """Caps applied equally to every arm. A lower cap is recorded as a deviation.

    ``max_examples`` keeps a prefix of canonical order. It is not a filter.
    ``walltime_s`` is checked between examples and accumulates across resumes.
    """

    max_examples: int | None = None
    max_samples: int | None = None
    max_new_tokens: int | None = None
    walltime_s: float | None = None


@dataclass(frozen=True)
class RunSpec:
    """Identity of one checkpoint under a published protocol.

    ``source_model`` is the base checkpoint the edit was applied to, for both
    arms. ``checkpoint_id`` distinguishes untouched from edited.
    """

    source_model: str
    checkpoint_id: str
    arm: str
    checkpoint_fingerprint: str
    source_revision: str | None = None
    protocol_name: str = "think35"
    engine: str | None = None
    dtype: str | None = None


def _dump(payload) -> str:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _digest(payload) -> str:
    return hashlib.sha256(_dump(payload).encode()).hexdigest()


def _problem_sha(problem: str) -> str:
    return hashlib.sha256(problem.encode()).hexdigest()


def _finite(value, label: str):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{label} must be a finite number")
    if value != value or value in (float("inf"), float("-inf")):
        raise ProtocolError(f"{label} must be finite")
    return value


def _resolve_limits(
    proto, dataset: str, budgets: Budgets
) -> tuple[int, int, list[str]]:
    if budgets.max_examples is not None and budgets.max_examples < 1:
        raise ProtocolError("max_examples must be at least 1")
    if budgets.walltime_s is not None and _finite(budgets.walltime_s, "walltime_s") < 0:
        raise ProtocolError("walltime_s must be >= 0")
    published_samples = proto.samples.get(dataset)
    if budgets.max_samples is None:
        if published_samples is None:
            raise ProtocolError(
                f"protocol {proto.name} publishes no sample count for {dataset!r}; set max_samples"
            )
        samples = published_samples
        sample_notes = []
    else:
        if budgets.max_samples < 1:
            raise ProtocolError("max_samples must be at least 1")
        if published_samples is not None and budgets.max_samples > published_samples:
            raise ProtocolError(
                f"max_samples {budgets.max_samples} exceeds the published {published_samples}"
            )
        samples = budgets.max_samples
        if published_samples is None:
            sample_notes = [
                f"samples={samples} for {dataset}; the protocol publishes no count"
            ]
        elif samples < published_samples:
            sample_notes = [
                f"samples={samples} is below the published {published_samples}"
            ]
        else:
            sample_notes = []
    published_tokens = proto.max_new_tokens
    if budgets.max_new_tokens is None:
        tokens = published_tokens
        token_notes = []
    else:
        if budgets.max_new_tokens < 1:
            raise ProtocolError("max_new_tokens must be at least 1")
        if budgets.max_new_tokens > published_tokens:
            raise ProtocolError(
                f"max_new_tokens {budgets.max_new_tokens} exceeds the published {published_tokens}"
            )
        tokens = budgets.max_new_tokens
        token_notes = (
            []
            if tokens == published_tokens
            else [f"max_new_tokens={tokens} is below the published {published_tokens}"]
        )
    return samples, tokens, sample_notes + token_notes


def _selected(examples: ExampleSet, budgets: Budgets) -> tuple[ExampleSet, str]:
    if budgets.max_examples is None or budgets.max_examples >= len(examples):
        return examples, "full_loaded_set"
    return examples.prefix(budgets.max_examples), "canonical_prefix"


def _dataset_block(examples: ExampleSet, selection: str) -> dict:
    provenance = examples.provenance
    return {
        "name": examples.name,
        "split": examples.split,
        "role": examples.role,
        "source": provenance.get("source"),
        "hf_repo": provenance.get("hf_repo"),
        "hf_config": provenance.get("hf_config"),
        "revision": provenance.get("revision"),
        "revision_pinned": provenance.get("revision_pinned"),
        "file_sha256": provenance.get("file_sha256"),
        "loaded_row_count": provenance.get(
            "full_row_count", provenance.get("row_count")
        ),
        "published_count": provenance.get("published_count"),
        "loaded_count_matches_published": provenance.get("count_matches_published"),
        "loaded_content_sha256": provenance.get("content_sha256"),
        "scored_ids": [item.id for item in examples.examples],
        "scored_content_sha256": _digest(
            [
                {"id": item.id, "problem": item.problem, "answer": item.answer}
                for item in examples.examples
            ]
        ),
        "selection": selection,
        "article_13gram_decontamination": provenance.get(
            "article_13gram_decontamination"
        ),
    }


def _sampling(proto, max_new_tokens: int) -> dict:
    sampling = dict(proto.sampling)
    sampling["max_tokens"] = max_new_tokens
    return sampling


def build_config(
    spec: RunSpec, examples: ExampleSet, budgets: Budgets, *, rendering: str
) -> dict:
    if examples.role != "evaluation":
        raise SplitIsolationError("run_evaluation refuses practice sets")
    if spec.arm not in ("untouched", "edited"):
        raise ProtocolError("arm must be 'untouched' or 'edited'")
    if (
        not spec.source_model
        or not spec.checkpoint_id
        or not spec.checkpoint_fingerprint
    ):
        raise ProtocolError(
            "source_model, checkpoint_id, and checkpoint_fingerprint are required"
        )
    proto = protocol(spec.protocol_name)
    selected, selection = _selected(examples, budgets)
    samples, tokens, deviations = _resolve_limits(proto, examples.name, budgets)
    grader = math_verify_status()
    wall = None if budgets.walltime_s is None else float(budgets.walltime_s)
    config = {
        "format": FORMAT_VERSION,
        "protocol": proto.name,
        "protocol_source": proto.source,
        "thinking": proto.thinking,
        "instruction": proto.instruction,
        "headline": proto.headline,
        "seed": proto.sampling.get("seed"),
        "published_sampling": dict(proto.sampling),
        "effective_sampling": _sampling(proto, tokens),
        "samples": samples,
        "max_new_tokens": tokens,
        "deviations": deviations,
        "budgets": {
            "max_examples": budgets.max_examples,
            "max_samples": budgets.max_samples,
            "max_new_tokens": budgets.max_new_tokens,
            "walltime_s": wall,
        },
        "dataset": _dataset_block(selected, selection),
        "source_model": spec.source_model,
        "source_revision": spec.source_revision,
        "checkpoint_id": spec.checkpoint_id,
        "checkpoint_fingerprint": spec.checkpoint_fingerprint,
        "arm": spec.arm,
        "engine": spec.engine,
        "dtype": spec.dtype,
        "prompt_rendering": rendering,
        "grader": {
            "primary": "math-verify",
            "available": grader["available"],
            "version": grader["version"],
            "article_version": grader["article_version"],
            "version_matches_article": grader["version_matches_article"],
            "secondary": "hendrycks_is_equiv",
        },
    }
    return config


def _diff_paths(left, right, prefix="") -> list[str]:
    if type(left) is not type(right):
        return [prefix or "<root>"]
    if isinstance(left, dict):
        paths = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                paths.append(path)
            else:
                paths.extend(_diff_paths(left[key], right[key], path))
        return paths
    if left != right:
        return [prefix or "<root>"]
    return []


def _identity(config: dict) -> dict:
    return {key: value for key, value in config.items() if key not in _ARM_FIELDS}


class ResultStore:
    def __init__(self, path, config: dict, examples: ExampleSet, *, writable: bool):
        self.path = Path(path)
        self.config = config
        self.config_hash = _digest(config)
        self.examples = {item.id: item for item in examples.examples}
        self.order = [item.id for item in examples.examples]
        self.samples_n = int(config["samples"])
        self.writable = writable
        self.samples: dict[tuple[str, int], dict] = {}
        self.invocations: list[dict] = []
        self._groups: list[str] = []
        self._handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "a+b" if self.writable else "rb")
        fcntl.flock(
            self._handle.fileno(), fcntl.LOCK_EX if self.writable else fcntl.LOCK_SH
        )
        self._load()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None
        return False

    def _load(self) -> None:
        self._handle.seek(0)
        raw = self._handle.read()
        if raw == b"":
            if not self.writable:
                raise IntegrityError(f"{self.path} is empty")
            self._write_locked(
                _dump(
                    {
                        "record": "config",
                        "config_hash": self.config_hash,
                        "config": self.config,
                    }
                )
                + "\n"
            )
            return
        good, partial = _complete_bytes(raw)
        records = [
            _decode_line(line, index)
            for index, line in enumerate(good.decode().splitlines(), start=1)
        ]
        if not records or records[0].get("record") != "config":
            raise IntegrityError(f"{self.path} does not start with a config record")
        stored = records[0].get("config")
        stored_hash = records[0].get("config_hash")
        if stored_hash != _digest(stored):
            raise IntegrityError(
                f"{self.path} config hash does not match its config record"
            )
        if stored_hash != self.config_hash:
            fields = _diff_paths(_identity(stored), _identity(self.config))
            arm = _diff_paths(
                {key: stored.get(key) for key in _ARM_FIELDS},
                {key: self.config.get(key) for key in _ARM_FIELDS},
            )
            raise ConfigMismatch(
                f"{self.path} is bound to a different run. "
                f"Identity fields: {fields or 'none'}. Arm fields: {arm or 'none'}."
            )
        self._consume(records[1:])
        if self.writable and (partial or self._trailing_incomplete()):
            self._rewrite_locked()

    def _consume(self, records: list[dict]) -> None:
        groups: list[str] = []
        for record in records:
            kind = record.get("record")
            if kind == "invocation":
                elapsed = record.get("elapsed_s")
                if (
                    not isinstance(elapsed, (int, float))
                    or isinstance(elapsed, bool)
                    or elapsed < 0
                ):
                    raise IntegrityError("invocation elapsed_s is invalid")
                self.invocations.append(record)
                continue
            if kind != "sample":
                raise IntegrityError(f"unknown record {kind!r}")
            example_id = record.get("example_id")
            sample_index = record.get("sample_index")
            if example_id not in self.examples:
                raise IntegrityError(
                    f"sample {example_id!r} is not in the configured set"
                )
            if not isinstance(sample_index, int) or isinstance(sample_index, bool):
                raise IntegrityError("sample_index is invalid")
            if not 0 <= sample_index < self.samples_n:
                raise IntegrityError(
                    f"sample_index {sample_index} is outside 0..{self.samples_n - 1}"
                )
            key = (example_id, sample_index)
            if key in self.samples:
                raise IntegrityError(f"duplicate sample {example_id}:{sample_index}")
            live = self.examples[example_id]
            if record.get("gold") != live.answer or record.get(
                "problem_sha256"
            ) != _problem_sha(live.problem):
                raise IntegrityError(
                    f"{example_id} does not match the dataset row bound to this file"
                )
            self.samples[key] = record
            if not groups or groups[-1] != example_id:
                groups.append(example_id)
        incomplete = [
            example_id
            for example_id in groups
            if not self._example_complete(example_id)
        ]
        if len(incomplete) > 1 or (incomplete and incomplete[0] != groups[-1]):
            raise IntegrityError(
                "an unfinished example is not the trailing example; the file was not written example by example"
            )
        self._groups = groups

    def _example_complete(self, example_id: str) -> bool:
        return all(
            (example_id, index) in self.samples for index in range(self.samples_n)
        )

    def _trailing_incomplete(self) -> bool:
        return bool(self._groups) and not self._example_complete(self._groups[-1])

    def _rewrite_locked(self) -> None:
        if self._trailing_incomplete():
            drop = self._groups[-1]
            self.samples = {
                key: value for key, value in self.samples.items() if key[0] != drop
            }
            self._groups = self._groups[:-1]
        lines = [
            _dump(
                {
                    "record": "config",
                    "config_hash": self.config_hash,
                    "config": self.config,
                }
            )
        ]
        for example_id in self.order:
            for index in range(self.samples_n):
                record = self.samples.get((example_id, index))
                if record is not None:
                    lines.append(_dump(record))
        for record in self.invocations:
            lines.append(_dump(record))
        self._write_locked("\n".join(lines) + "\n")

    def _write_locked(self, text: str) -> None:
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(text.encode())
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def spent_walltime(self) -> float:
        return float(sum(record["elapsed_s"] for record in self.invocations))

    def pending(self) -> list:
        return [
            self.examples[example_id]
            for example_id in self.order
            if not self._example_complete(example_id)
        ]

    def append_example(self, records: list[dict]) -> None:
        if not records:
            raise IntegrityError("refusing to append an empty example")
        example_id = records[0]["example_id"]
        if any(record["example_id"] != example_id for record in records):
            raise IntegrityError("one append must cover a single example")
        indexes = [record["sample_index"] for record in records]
        if indexes != list(range(self.samples_n)):
            raise IntegrityError("samples must be appended together, in index order")
        payload = "\n".join(_dump(record) for record in records) + "\n"
        self._handle.seek(0, os.SEEK_END)
        self._handle.write(payload.encode())
        self._handle.flush()
        os.fsync(self._handle.fileno())
        for record in records:
            self.samples[(record["example_id"], record["sample_index"])] = record

    def append_invocation(self, elapsed_s: float, stopped_reason: str) -> None:
        record = {
            "record": "invocation",
            "elapsed_s": elapsed_s,
            "stopped_reason": stopped_reason,
        }
        self._handle.seek(0, os.SEEK_END)
        self._handle.write((_dump(record) + "\n").encode())
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self.invocations.append(record)

    def completed_ids(self) -> list[str]:
        return [
            example_id
            for example_id in self.order
            if self._example_complete(example_id)
        ]

    def sample_rows(self) -> list[dict]:
        rows = []
        for example_id in self.completed_ids():
            for index in range(self.samples_n):
                rows.append(self.samples[(example_id, index)])
        return rows

    def summary(self) -> dict:
        return summarize_run(self.config, self.sample_rows(), self.invocations)


def _complete_bytes(raw: bytes) -> tuple[bytes, bool]:
    if raw.endswith(b"\n"):
        return raw, False
    newline = raw.rfind(b"\n")
    if newline < 0:
        return b"", True
    return raw[: newline + 1], True


def _decode_line(line: str, number: int) -> dict:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise IntegrityError(f"line {number} is not JSON") from error
    if not isinstance(record, dict):
        raise IntegrityError(f"line {number} is not an object")
    return record


def read_run(path) -> dict:
    """Read a results file without creating or truncating it."""

    file_path = Path(path)
    with open(file_path, "rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            raw = handle.read()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    good, partial = _complete_bytes(raw)
    if good == b"":
        raise IntegrityError(f"{file_path} has no complete record")
    records = [
        _decode_line(line, index)
        for index, line in enumerate(good.decode().splitlines(), start=1)
    ]
    if records[0].get("record") != "config":
        raise IntegrityError(f"{file_path} does not start with a config record")
    config = records[0]["config"]
    if records[0].get("config_hash") != _digest(config):
        raise IntegrityError(
            f"{file_path} config hash does not match its config record"
        )
    samples: dict[tuple[str, int], dict] = {}
    invocations = []
    groups: list[str] = []
    samples_n = config["samples"]
    allowed = set(config["dataset"]["scored_ids"])
    for record in records[1:]:
        if record.get("record") == "invocation":
            invocations.append(record)
            continue
        if record.get("record") != "sample":
            raise IntegrityError(f"unknown record {record.get('record')!r}")
        example_id = record["example_id"]
        sample_index = record["sample_index"]
        if (
            example_id not in allowed
            or not isinstance(sample_index, int)
            or not 0 <= sample_index < samples_n
        ):
            raise IntegrityError(
                f"sample {example_id}:{sample_index} is outside the configured set"
            )
        key = (example_id, sample_index)
        if key in samples:
            raise IntegrityError(f"duplicate sample {example_id}:{sample_index}")
        samples[key] = record
        if not groups or groups[-1] != example_id:
            groups.append(example_id)
    incomplete = [
        example_id
        for example_id in groups
        if any((example_id, index) not in samples for index in range(samples_n))
    ]
    if len(incomplete) > 1 or (incomplete and groups and incomplete[0] != groups[-1]):
        raise IntegrityError(f"{file_path} has a non-trailing unfinished example")
    if incomplete:
        drop = incomplete[0]
        samples = {key: value for key, value in samples.items() if key[0] != drop}
    completed = [
        example_id
        for example_id in config["dataset"]["scored_ids"]
        if all((example_id, index) in samples for index in range(samples_n))
    ]
    rows = [
        samples[(example_id, index)]
        for example_id in completed
        for index in range(samples_n)
    ]
    return {
        "path": str(file_path),
        "config": config,
        "rows": rows,
        "invocations": invocations,
        "completed_ids": completed,
        "trailing_partial": partial,
        "summary": summarize_run(config, rows, invocations),
    }


def _mean(values: list) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return sum(values) / len(values)


def summarize_run(config: dict, rows: list[dict], invocations: list[dict]) -> dict:
    configured = config["dataset"]["scored_ids"]
    samples_n = config["samples"]
    by_id: dict[str, list] = {example_id: [] for example_id in configured}
    for row in rows:
        by_id[row["example_id"]].append(row)
    completed = [
        example_id for example_id in configured if len(by_id[example_id]) == samples_n
    ]
    ordered = [
        row
        for example_id in completed
        for row in sorted(by_id[example_id], key=lambda item: item["sample_index"])
    ]
    acc = _mean([row["correct"] for row in ordered])
    acc_equiv = (
        _mean([1.0 if row["equiv"] else 0.0 for row in ordered]) if ordered else None
    )
    acc_lenient = _mean([row["lenient_correct"] for row in ordered])
    cut_share = _mean(
        [1.0 if row["cut"] else 0.0 if row["cut"] is False else None for row in ordered]
    )
    verbatim = (
        _mean([1.0 if row["loop"]["verbatim_loop_end"] else 0.0 for row in ordered])
        if ordered
        else None
    )
    repetitive = (
        _mean([1.0 if row["loop"]["repetitive_end"] else 0.0 for row in ordered])
        if ordered
        else None
    )
    mean_tokens = _mean([row["n_tokens"] for row in ordered])
    no_answer = _mean(
        [
            1.0
            if row["no_answer_finished"]
            else 0.0
            if row["no_answer_finished"] is False
            else None
            for row in ordered
        ]
    )
    stopped = invocations[-1]["stopped_reason"] if invocations else None
    return {
        "protocol": config["protocol"],
        "arm": config["arm"],
        "checkpoint_id": config["checkpoint_id"],
        "checkpoint_fingerprint": config["checkpoint_fingerprint"],
        "source_model": config["source_model"],
        "source_revision": config["source_revision"],
        "dataset": config["dataset"]["name"],
        "split": config["dataset"]["split"],
        "n_problems": len(completed),
        "n_configured": len(configured),
        "samples": samples_n,
        "complete": len(completed) == len(configured) and len(configured) > 0,
        "stopped_reason": stopped,
        "spent_walltime_s": sum(item["elapsed_s"] for item in invocations),
        "acc": acc,
        "acc_equiv": acc_equiv,
        "acc_lenient": acc_lenient,
        "headline": config["headline"],
        "headline_value": {"acc": acc, "acc_lenient": acc_lenient}.get(
            config["headline"]
        ),
        "cut_share": cut_share,
        "mean_tokens": mean_tokens,
        "no_answer_finished_share": no_answer,
        "verbatim_loop_end_share": verbatim,
        "repetitive_end_share": repetitive,
        "grader": config["grader"],
        "deviations": list(config["deviations"]),
        "metrics_are_article_grader": acc is not None
        and config["grader"]["version_matches_article"],
        "completed_ids": completed,
    }


def _requests(
    example, spec_protocol, samples: int, sampling: dict, rendered: str | None
) -> list[dict]:
    body = user_turn(example.problem, instruction=spec_protocol.instruction)
    requests = []
    for index in range(samples):
        requests.append(
            {
                "example_id": example.id,
                "sample_index": index,
                "index": example.index,
                "user_content": body,
                "rendered_prompt": rendered,
                "sampling": dict(sampling),
                "max_new_tokens": sampling["max_tokens"],
                "seed": sampling.get("seed"),
                "protocol": spec_protocol.name,
            }
        )
    return requests


def _check_request_closed(request: dict) -> None:
    leaked = sorted({"answer", "gold", "solution"} & set(request))
    if leaked:
        raise IntegrityError(f"generation request leaks {leaked}")


def _grade_example(example, responses: list[dict], requests: list[dict]) -> list[dict]:
    if len(responses) != len(requests):
        raise GenerationRejected(
            f"{example.id}: generator returned {len(responses)} rows for {len(requests)} requests"
        )
    records = []
    for request, response in zip(requests, responses):
        if not isinstance(response, dict) or not isinstance(response.get("text"), str):
            raise GenerationRejected(f"{example.id}: each response needs a text string")
        if response.get("example_id", request["example_id"]) != request["example_id"]:
            raise GenerationRejected(
                f"{example.id}: response example_id does not match the request"
            )
        finish = response.get("finish_reason")
        if finish not in _ACCEPTED_FINISH:
            raise GenerationRejected(
                f"{example.id} sample {request['sample_index']}: finish_reason {finish!r} "
                "is not 'stop' or 'length'; the sample was not stored"
            )
        n_tokens = response.get("n_tokens")
        graded = grade_response(
            response["text"],
            example.answer,
            finish_reason=finish,
            n_tokens=n_tokens,
        )
        records.append(
            {
                "record": "sample",
                "example_id": example.id,
                "sample_index": request["sample_index"],
                "index": example.index,
                "problem_sha256": _problem_sha(example.problem),
                "gold": example.answer,
                "text": response["text"],
                **graded,
            }
        )
    return records


def _resolve_rendering(tokenizer, render, thinking: bool, instruction: str):
    if tokenizer is not None and render is not None:
        raise ProtocolError("pass tokenizer or render, not both")
    if tokenizer is not None:

        def _render(problem: str) -> str:
            from .protocol import render_chat

            return render_chat(
                tokenizer, problem, thinking=thinking, instruction=instruction
            )

        return "apply_chat_template", _render
    if render is not None:
        return "custom", render
    return "generator", None


def run_evaluation(
    examples,
    spec: RunSpec,
    generate,
    out_path,
    *,
    budgets: Budgets | None = None,
    tokenizer=None,
    render=None,
    clock=None,
) -> dict:
    """Grade ``examples`` with ``generate`` and append to a config-bound JSONL file.

    ``generate`` receives one example's requests and returns one response per
    request, in the same order. A response is ``text``, ``finish_reason``
    (``stop`` or ``length``), and optional ``n_tokens``. Completed examples are
    skipped on the next call with the same spec, dataset, and budgets.
    """

    if not isinstance(examples, ExampleSet):
        raise TypeError("examples must be an ExampleSet from load_jsonl or load_hf")
    budgets = budgets or Budgets()
    proto = protocol(spec.protocol_name)
    rendering, render_fn = _resolve_rendering(
        tokenizer, render, proto.thinking, proto.instruction
    )
    config = build_config(spec, examples, budgets, rendering=rendering)
    selected, _selection = _selected(examples, budgets)
    clock = clock or time.monotonic
    with ResultStore(out_path, config, selected, writable=True) as store:
        spent = store.spent_walltime()
        started = clock()
        stopped = "completed"
        wall = config["budgets"]["walltime_s"]
        work_left = bool(store.pending())
        try:
            for example in store.pending():
                now = clock()
                if wall is not None and spent + (now - started) >= wall:
                    stopped = "walltime"
                    break
                rendered = None if render_fn is None else render_fn(example.problem)
                requests = _requests(
                    example,
                    proto,
                    config["samples"],
                    config["effective_sampling"],
                    rendered,
                )
                for request in requests:
                    _check_request_closed(request)
                responses = generate(requests)
                store.append_example(_grade_example(example, responses, requests))
        except GenerationRejected:
            stopped = "generation_rejected"
            raise
        except Exception:
            stopped = "error"
            raise
        finally:
            if work_left:
                store.append_invocation(float(clock() - started), stopped)
        return store.summary()


def _blog_blockers(config: dict, summary: dict) -> list[str]:
    blockers = []
    dataset = config["dataset"]
    if config["protocol"] != "think35":
        blockers.append("protocol is not think35, the blog table profile")
    if (
        dataset["name"] not in ("math500", "gsm8k")
        or dataset["split"] != "test"
        or dataset["role"] != "evaluation"
    ):
        blockers.append("dataset is not MATH-500 test or GSM8K test")
    if dataset.get("source") != "huggingface":
        blockers.append(
            "dataset was a local JSONL, not the published Hugging Face split"
        )
    expected_repo = {"math500": "HuggingFaceH4/MATH-500", "gsm8k": "openai/gsm8k"}.get(
        dataset["name"]
    )
    if (
        dataset.get("source") == "huggingface"
        and dataset.get("hf_repo") != expected_repo
    ):
        blockers.append("Hugging Face repo does not match the published loader")
    if dataset.get("selection") != "full_loaded_set":
        blockers.append("scored set is a prefix, not the full loaded split")
    published = dataset.get("published_count")
    if (
        published is None
        or dataset.get("loaded_row_count") != published
        or summary["n_configured"] != published
    ):
        blockers.append("row count is not the published full split")
    if not summary["complete"]:
        blockers.append("the configured set is not fully graded")
    if config["deviations"]:
        blockers.append("sampling or sample count deviates from the published protocol")
    grader = config["grader"]
    if not grader["available"] or not grader["version_matches_article"]:
        blockers.append(
            f"primary grader is not math-verify {grader['article_version']}"
        )
    if not summary["metrics_are_article_grader"]:
        blockers.append("accuracy is null or was not produced by math-verify")
    if summary["n_problems"] and summary["mean_tokens"] is None:
        blockers.append("token counts are missing")
    if summary["n_problems"] and summary["cut_share"] is None:
        blockers.append("cut_share is unknown")
    if config["prompt_rendering"] != "apply_chat_template":
        blockers.append("the chat template was not applied with enable_thinking")
    if config["engine"] != "vllm":
        blockers.append("engine is not declared as vllm")
    if config["dtype"] != "bfloat16":
        blockers.append("dtype is not declared as bfloat16")
    if config["source_model"] != QWEN35_BASE_MODEL:
        blockers.append("source_model is not Qwen/Qwen3.5-0.8B")
    if config["source_revision"] != QWEN35_BASE_REVISION:
        blockers.append("source_revision is not the published Qwen3.5-0.8B revision")
    return blockers


def _notes(config: dict) -> list[str]:
    notes = [
        "cut_share is the blog hit-token-limit column (finish_reason == length).",
        "verbatim_loop_end is mathqa.loop_signals. repetitive_end follows that docstring; mcqa.loop_signals is not public.",
        "acc_equiv is Hendrycks is_equiv, not the article accuracy.",
        "Engine, dtype, and the vLLM constructor seed are caller-declared and are not observed.",
        "Dataset revision pinning is not in the published loader.",
        "The 13-word practice filter is not applied.",
    ]
    if config["dataset"].get("revision_pinned") is False:
        notes.append("This run did not pin a dataset revision.")
    return notes


def compare_runs(untouched_path, edited_path) -> dict:
    """Compare two result files. A delta is set only when the runs match and both are complete.

    Checkpoint id, fingerprint, and arm may differ. Budgets, dataset bytes, protocol,
    grader, and the scored ids may not. A hole in the id list is treated as selection
    and is not scored as a delta.
    """

    untouched = read_run(untouched_path)
    edited = read_run(edited_path)
    left = untouched["config"]
    right = edited["config"]
    mismatches = _diff_paths(_identity(left), _identity(right))
    reasons = []
    if mismatches:
        reasons.append("config_mismatch:" + ",".join(mismatches))
    if left["arm"] != "untouched" or right["arm"] != "edited":
        reasons.append("arm labels must be untouched and edited on the two paths")
    if left["checkpoint_id"] == right["checkpoint_id"]:
        reasons.append("checkpoint ids are the same")
    if untouched["completed_ids"] != left["dataset"]["scored_ids"]:
        reasons.append("untouched run is incomplete")
    if edited["completed_ids"] != right["dataset"]["scored_ids"]:
        reasons.append("edited run is incomplete")
    if untouched["completed_ids"] != edited["completed_ids"]:
        reasons.append("completed example ids differ")
    for label, run in (("untouched", untouched), ("edited", edited)):
        done = run["completed_ids"]
        configured_ids = run["config"]["dataset"]["scored_ids"]
        if done != configured_ids[: len(done)]:
            reasons.append(f"{label} completed ids are not a canonical prefix")
    comparable = not reasons
    complete = (
        comparable
        and untouched["summary"]["complete"]
        and edited["summary"]["complete"]
    )
    delta = None
    if (
        complete
        and untouched["summary"]["acc"] is not None
        and edited["summary"]["acc"] is not None
    ):
        delta = edited["summary"]["acc"] - untouched["summary"]["acc"]
    per_example = None
    if comparable:
        per_example = _paired(untouched, edited)
    blockers = _blog_blockers(left, untouched["summary"]) + [
        "edited:" + item for item in _blog_blockers(right, edited["summary"])
    ]
    return {
        "comparable": comparable,
        "complete": complete,
        "incomparable_because": reasons,
        "delta_acc": delta,
        "blog_table_settings": comparable and complete and not blockers,
        "blog_table_blockers": blockers,
        "notes": _notes(left),
        "untouched": untouched["summary"],
        "edited": edited["summary"],
        "per_example": per_example,
        "checkpoint_fingerprints_match": left["checkpoint_fingerprint"]
        == right["checkpoint_fingerprint"],
    }


def _paired(untouched: dict, edited: dict) -> list[dict]:
    def rows_for(run):
        grouped: dict[str, list] = {}
        for row in run["rows"]:
            grouped.setdefault(row["example_id"], []).append(row)
        return grouped

    left = rows_for(untouched)
    right = rows_for(edited)
    paired = []
    for example_id in untouched["completed_ids"]:
        a = sorted(left[example_id], key=lambda item: item["sample_index"])
        b = sorted(right[example_id], key=lambda item: item["sample_index"])
        if [row["gold"] for row in a] != [row["gold"] for row in b]:
            raise IntegrityError(f"{example_id} gold differs between arms")
        paired.append(
            {
                "id": example_id,
                "untouched_correct": [row["correct"] for row in a],
                "edited_correct": [row["correct"] for row in b],
                "untouched_cut": [row["cut"] for row in a],
                "edited_cut": [row["cut"] for row in b],
                "untouched_equiv": [row["equiv"] for row in a],
                "edited_equiv": [row["equiv"] for row in b],
            }
        )
    return paired


def write_summary(path, summary: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
    )
    os.replace(temporary, target)
