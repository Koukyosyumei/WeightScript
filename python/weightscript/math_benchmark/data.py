"""MATH-500 and GSM8K loading.

Evaluation rows come from the published test splits. Practice rows come from
the GSM8K train split or from a caller-built JSONL. The Hugging Face path is
optional and runs only when asked. A local JSONL is the path that does not
download anything.

The public Experiment 8 code drops a practice problem when a 13-word window or
its normalized text occurs in an evaluation problem (``practice.grams`` /
``practice.norm``). Those functions are not in the public release. This module
does not invent that filter. ``assert_disjoint`` checks exact normalized text
and says so.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .extract import gold_of_solution, gsm8k_reference_answer
from .protocol import GSM8K_HF, MATH500_HF, PUBLISHED_COUNTS


class SplitIsolationError(ValueError):
    """A split or role would mix practice data with the reported benchmark."""


@dataclass(frozen=True)
class Example:
    id: str
    problem: str
    answer: str
    index: int

    def to_json(self) -> dict:
        return {"id": self.id, "problem": self.problem, "answer": self.answer, "index": self.index}


@dataclass(frozen=True)
class ExampleSet:
    name: str
    split: str
    role: str
    examples: tuple[Example, ...]
    provenance: dict

    def __len__(self) -> int:
        return len(self.examples)

    def prefix(self, count: int | None) -> "ExampleSet":
        """Canonical prefix. This is the only subset the runner will score."""

        if count is None or count >= len(self.examples):
            return self
        if count < 1:
            raise ValueError("max_examples must be at least 1")
        selected = self.examples[:count]
        provenance = dict(self.provenance)
        provenance["selection"] = "canonical_prefix"
        provenance["prefix_count"] = count
        provenance["full_row_count"] = len(self.examples)
        return ExampleSet(self.name, self.split, self.role, selected, provenance)


def normalize_problem(text: str) -> str:
    """Whitespace-collapsed casefolded NFKC. Not the unpublished practice.norm."""

    folded = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", folded).strip()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _content_hash(examples: tuple[Example, ...]) -> str:
    lines = [
        json.dumps({"id": item.id, "problem": item.problem, "answer": item.answer}, ensure_ascii=False, sort_keys=True)
        for item in examples
    ]
    return _sha256_bytes(("\n".join(lines) + "\n").encode())


def _check_role(dataset: str, split: str, role: str) -> None:
    if role not in ("evaluation", "practice"):
        raise SplitIsolationError("role must be 'evaluation' or 'practice'")
    if dataset == "math500" and (role != "evaluation" or split != "test"):
        raise SplitIsolationError("MATH-500 is an evaluation set: role=evaluation and split=test")
    if dataset == "gsm8k" and role == "evaluation" and split != "test":
        raise SplitIsolationError("GSM8K evaluation uses the test split only")
    if dataset == "gsm8k" and role == "practice" and split != "train":
        raise SplitIsolationError("GSM8K practice uses the train split only")
    if dataset == "math_train" and (role != "practice" or split != "train"):
        raise SplitIsolationError("MATH train is practice only and is not an evaluation set")
    if role == "evaluation" and split == "train":
        raise SplitIsolationError("an evaluation run cannot use a train split")
    if role == "practice" and split == "test":
        raise SplitIsolationError("a practice set cannot use a test split")


def _answer_for(dataset: str, raw: str, solution: str | None) -> str:
    if dataset == "gsm8k":
        return gsm8k_reference_answer(raw)
    if raw.strip() == "" and solution:
        extracted = gold_of_solution(solution)
        if not extracted:
            raise ValueError("MATH solution has no closed boxed answer")
        return extracted
    if raw.strip() == "":
        raise ValueError("answer is empty")
    return raw


def _build(name: str, split: str, role: str, rows: list[dict], provenance: dict) -> ExampleSet:
    _check_role(name, split, role)
    examples = []
    seen = set()
    for index, row in enumerate(rows):
        example_id = str(row["id"])
        if example_id in seen:
            raise ValueError(f"duplicate example id {example_id!r}")
        seen.add(example_id)
        problem = row["problem"]
        if not isinstance(problem, str) or problem.strip() == "":
            raise ValueError(f"{example_id}: problem is empty")
        answer = _answer_for(name, str(row.get("answer", "")), row.get("solution"))
        if answer.strip() == "":
            raise ValueError(f"{example_id}: answer is empty")
        examples.append(Example(example_id, problem, answer, index))
    packed = tuple(examples)
    expected = PUBLISHED_COUNTS.get((name, split))
    provenance = dict(provenance)
    provenance.update(
        {
            "dataset": name,
            "split": split,
            "role": role,
            "row_count": len(packed),
            "published_count": expected,
            "count_matches_published": expected == len(packed) if expected is not None else None,
            "content_sha256": _content_hash(packed),
            "example_ids_sha256": _sha256_bytes(
                json.dumps([item.id for item in packed], ensure_ascii=False).encode()
            ),
            "article_13gram_decontamination": "not_applied_filter_unpublished",
            "order": "source_order",
        }
    )
    return ExampleSet(name, split, role, packed, provenance)


def load_jsonl(path, *, dataset: str, split: str, role: str) -> ExampleSet:
    """Load a local JSONL file of ``id``, ``problem``, and ``answer`` objects.

    GSM8K answers may still be the original field with a ``####`` marker. MATH
    practice rows may carry ``solution`` instead of ``answer``. No network call.
    """

    file_path = Path(path)
    raw = file_path.read_bytes()
    rows = []
    for line_number, line in enumerate(raw.decode().splitlines(), start=1):
        if line.strip() == "":
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{file_path}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{file_path}:{line_number}: expected a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{file_path} has no examples")
    return _build(
        dataset,
        split,
        role,
        rows,
        {
            "source": "local_jsonl",
            "path": str(file_path),
            "file_sha256": _sha256_bytes(raw),
            "hf_repo": None,
            "hf_config": None,
            "revision": None,
            "revision_pinned": False,
        },
    )


def load_hf(dataset: str, *, split: str, revision: str | None = None, cache_dir: str | None = None):
    """Load a published Hugging Face split. Imports ``datasets`` only here.

    ``dataset`` is ``math500`` or ``gsm8k``. MATH-500 accepts only ``test``.
    GSM8K accepts ``test`` (evaluation) or ``train`` (practice). The article
    loader does not pin a revision; pass one only when you have it.
    """

    if dataset == "math500":
        role = "evaluation"
        repo = MATH500_HF["repo"]
        config = MATH500_HF["config"]
        if split != "test":
            raise SplitIsolationError("MATH-500 publishes a test split only")
    elif dataset == "gsm8k":
        repo = GSM8K_HF["repo"]
        config = GSM8K_HF["config"]
        role = "evaluation" if split == "test" else "practice"
    else:
        raise ValueError("load_hf supports math500 and gsm8k; use load_jsonl for other files")
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as error:
        raise ImportError(
            "Loading Hugging Face splits requires the optional 'datasets' package. "
            "Pass a local JSONL to load_jsonl to avoid that download."
        ) from error
    kwargs = {"split": split}
    if revision is not None:
        kwargs["revision"] = revision
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if config is None:
        loaded = load_dataset(repo, **kwargs)
    else:
        loaded = load_dataset(repo, config, **kwargs)
    rows = []
    for index, row in enumerate(loaded):
        if dataset == "math500":
            rows.append(
                {"id": row["unique_id"], "problem": row["problem"], "answer": row["answer"]}
            )
        else:
            rows.append(
                {
                    "id": f"gsm8k_{split}_{index}",
                    "problem": row["question"],
                    "answer": row["answer"],
                }
            )
    fingerprint = getattr(loaded, "_fingerprint", None)
    return _build(
        dataset,
        split,
        role,
        rows,
        {
            "source": "huggingface",
            "path": None,
            "file_sha256": None,
            "hf_repo": repo,
            "hf_config": config,
            "revision": revision,
            "revision_pinned": revision is not None,
            "datasets_fingerprint": fingerprint,
        },
    )


def assert_disjoint(evaluation: ExampleSet, practice: ExampleSet) -> None:
    """Reject exact normalized-text overlap. Does not apply the unpublished 13-gram filter."""

    if evaluation.role != "evaluation":
        raise SplitIsolationError("the first set must be an evaluation set")
    if practice.role != "practice":
        raise SplitIsolationError("the second set must be a practice set")
    practice_text = {}
    for item in practice.examples:
        practice_text.setdefault(normalize_problem(item.problem), item.id)
    overlaps = []
    for item in evaluation.examples:
        key = normalize_problem(item.problem)
        if key in practice_text:
            overlaps.append((item.id, practice_text[key]))
    if overlaps:
        shown = ", ".join(f"{eval_id}~{practice_id}" for eval_id, practice_id in overlaps[:5])
        raise SplitIsolationError(
            f"{len(overlaps)} evaluation problem(s) match practice text after NFKC/casefold "
            f"normalization ({shown}). This is not the article's 13-word filter."
        )


def write_jsonl(examples: ExampleSet, path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(item.to_json(), ensure_ascii=False, sort_keys=True) for item in examples.examples]
    file_path.write_text("\n".join(lines) + "\n")
