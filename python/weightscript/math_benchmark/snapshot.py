"""Save a published MATH-500 or GSM8K split as local JSONL.

The article loaders do not pin a dataset revision. A snapshot records the URL
and any commit header the server sends, with ``revision_pinned`` false. The
JSONL is a local file. Loading it does not count as the Hugging Face source,
so ``blog_table_settings`` stays false until ``load_hf`` is used.

GSM8K on Hugging Face is parquet. Reading it imports ``pyarrow`` only inside
the conversion. MATH-500 ``test.jsonl`` needs no extra package.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from .data import dump_jsonl_line, load_jsonl
from .locations import dataset_cache_dir
from .protocol import GSM8K_HF, MATH500_HF, PUBLISHED_COUNTS

MAX_PUBLISHED_BYTES = 32 * 1024 * 1024
_USER_AGENT = "weightscript-math-benchmark"


def _hf_url(dataset: str, split: str) -> tuple[str, str]:
    if dataset == "math500":
        if split != "test":
            raise ValueError("MATH-500 publishes a test split only")
        repo = MATH500_HF["repo"]
        return f"https://huggingface.co/datasets/{repo}/resolve/main/test.jsonl", "jsonl"
    if dataset == "gsm8k":
        if split not in ("test", "train"):
            raise ValueError("GSM8K snapshot splits are test and train")
        repo = GSM8K_HF["repo"]
        config = GSM8K_HF["config"]
        name = f"{split}-00000-of-00001.parquet"
        return f"https://huggingface.co/datasets/{repo}/resolve/main/{config}/{name}", "parquet"
    raise ValueError("snapshot supports math500 and gsm8k")


def math500_rows(text: str) -> list[dict]:
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.strip() == "":
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"MATH-500 line {line_number} is not an object")
        rows.append(
            {
                "id": str(raw["unique_id"]),
                "problem": raw["problem"],
                "answer": raw["answer"],
            }
        )
    return rows


def gsm8k_rows(records, split: str) -> list[dict]:
    rows = []
    for index, raw in enumerate(records):
        if not isinstance(raw, dict):
            raise ValueError(f"GSM8K row {index} is not an object")
        rows.append(
            {
                "id": f"gsm8k_{split}_{index}",
                "problem": raw["question"],
                "answer": raw["answer"],
            }
        )
    return rows


def rows_from_parquet(path) -> list[dict]:
    """Read ``question`` and ``answer`` columns. Imports pyarrow only here."""

    try:
        import pyarrow.parquet as parquet  # noqa: PLC0415
    except ImportError as error:
        raise ImportError(
            "The published GSM8K files are parquet. Install pyarrow to convert them, "
            "or pass a JSONL of question/answer rows in dataset order to load_jsonl."
        ) from error
    table = parquet.read_table(Path(path), columns=["question", "answer"])
    return table.to_pylist()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _download(url: str, dest: Path, opener) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with opener(request, timeout=60) as response:
            status = getattr(response, "status", None)
            headers = {key.lower(): value for key, value in response.headers.items()}
            length = headers.get("content-length")
            if length is not None and int(length) > MAX_PUBLISHED_BYTES:
                raise ValueError(f"{url} announces {length} bytes, above {MAX_PUBLISHED_BYTES}")
            chunks = []
            total = 0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > MAX_PUBLISHED_BYTES:
                    raise ValueError(f"{url} exceeded {MAX_PUBLISHED_BYTES} bytes")
                chunks.append(block)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    payload = b"".join(chunks)
    partial.write_bytes(payload)
    partial.replace(dest)
    return {
        "url": url,
        "http_status": status,
        "observed_repo_commit": headers.get("x-repo-commit"),
        "revision_pinned": False,
        "article_pins_dataset_revision": False,
        "raw_sha256": _sha256(payload),
        "raw_bytes": len(payload),
    }


def fetch_split(dataset: str, split: str, dest=None, *, opener=None, check_count: bool = True) -> dict:
    """Download one published split into ``dest`` and return its provenance.

    ``dest`` is required unless ``dataset_cache_dir()`` resolves a directory.
    Nothing is written when that directory is unset.
    """

    directory = dest if dest is not None else dataset_cache_dir()
    if directory is None:
        raise ValueError("pass dest or set WEIGHTSCRIPT_CACHE_DIR or cache_dir")
    root = Path(directory)
    url, kind = _hf_url(dataset, split)
    folder = root / dataset
    raw_path = folder / f"{split}.raw.{kind}"
    jsonl_path = folder / f"{split}.jsonl"
    provenance_path = folder / f"{split}.provenance.json"
    opener = opener or urllib.request.urlopen
    remote = _download(url, raw_path, opener)
    pending = {
        **remote,
        "dataset": dataset,
        "split": split,
        "format": kind,
        "raw_path": str(raw_path),
        "jsonl_path": None,
        "scores_as_published_hf_split": False,
        "note": "Local JSONL is not load_hf. blog_table_settings stays false for this file.",
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(pending, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    try:
        if kind == "jsonl":
            rows = math500_rows(raw_path.read_text())
        else:
            rows = gsm8k_rows(rows_from_parquet(raw_path), split)
    except Exception as error:
        pending["convert_error"] = f"{type(error).__name__}: {error}"
        provenance_path.write_text(json.dumps(pending, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
        raise
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text("".join(dump_jsonl_line(row) + "\n" for row in rows))
    role = "practice" if dataset == "gsm8k" and split == "train" else "evaluation"
    loaded = load_jsonl(jsonl_path, dataset=dataset, split=split, role=role)
    expected = PUBLISHED_COUNTS.get((dataset, split))
    provenance = {
        **remote,
        "dataset": dataset,
        "split": split,
        "role": role,
        "format": kind,
        "raw_path": str(raw_path),
        "jsonl_path": str(jsonl_path),
        "jsonl_sha256": _sha256(jsonl_path.read_bytes()),
        "row_count": len(loaded),
        "published_count": expected,
        "count_matches_published": expected == len(loaded) if expected is not None else None,
        "content_sha256": loaded.provenance["content_sha256"],
        "scores_as_published_hf_split": False,
        "note": "Local JSONL is not load_hf. blog_table_settings stays false for this file.",
    }
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    if check_count and expected is not None and len(loaded) != expected:
        raise ValueError(f"{dataset} {split} has {len(loaded)} rows, published count is {expected}")
    return provenance
