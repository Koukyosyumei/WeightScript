"""Answer extraction and grading for the Experiment 8 math harness.

The boxed-answer reader and ``is_equiv`` are the MIT-licensed grader shipped with
the public logs, which itself carries the MIT-licensed Hendrycks MATH checker.
Primary equivalence is math-verify when that package is installed. This module
does not replace math-verify with ``is_equiv``.

Copyright (c) 2026 the authors of "LLMs can understand and write neural network weights"

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in the
Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The Hendrycks functions ``last_boxed_only_string``, ``remove_boxed``,
``_strip_string`` and ``is_equiv`` are from hendrycks/math (MIT), commit
985bdc16, via exp8_math/code/mathqa.py. The bare ``except`` clauses in that port
are ``except Exception``.

math-verify is imported on first use. mathqa.py calls
``parse('\\\\boxed{' + answer + '}')`` and ``verify(gold, pred)`` from
math-verify 0.9.0.
"""

from __future__ import annotations

import re
import threading

from .loops import loop_signals
from .protocol import MATH_VERIFY_VERSION


_BOX_FORMS = re.compile(r"\\boxed|\\fbox")
_MV = {"module": None, "error": None, "loaded": False}


def answer_text(text: str) -> str:
    """Drop a closed reasoning block. The last ``</think>`` ends it."""

    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


def last_boxed_only_string(string: str):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:  # noqa: BLE001
        return None


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:  # noqa: BLE001
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:  # noqa: BLE001
        return string


def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def is_equiv(str1, str2) -> bool:
    """Hendrycks MATH string equivalence. Not the Experiment 8 primary score."""

    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        return _strip_string(str1) == _strip_string(str2)
    except Exception:  # noqa: BLE001
        return str1 == str2


def _box_at(text: str, idx: int):
    key = "\\boxed" if text.startswith("\\boxed", idx) else "\\fbox"
    rest = text[idx + len(key) :]
    stripped = rest.lstrip()
    if stripped.startswith("{"):
        depth = 0
        for k, ch in enumerate(stripped):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out = stripped[1:k].strip()
                    return out or None
        return None
    if key == "\\boxed" and rest.startswith(" "):
        out = rest.strip().split("$")[0].split("\n")[0].strip()
        return out or None
    return None


def last_boxed(text: str):
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    return _box_at(text, idx)


def extract_answer(text: str) -> tuple[str | None, int]:
    """``(inner answer or None, box count)`` on the text after the last ``</think>``."""

    visible = answer_text(text)
    return last_boxed(visible), len(_BOX_FORMS.findall(visible))


def gold_of_solution(solution: str) -> str | None:
    """Gold answer of a MATH reference solution: its last box."""

    return last_boxed(solution)


def gsm8k_reference_answer(answer_field: str) -> str:
    """GSM8K gold as evaluate.py prepares it: text after ``####``, commas removed."""

    if not isinstance(answer_field, str) or answer_field.strip() == "":
        raise ValueError("GSM8K answer field is empty")
    return answer_field.split("####")[-1].strip().replace(",", "")


def _load_math_verify():
    if _MV["loaded"]:
        return _MV["module"], _MV["error"]
    _MV["loaded"] = True
    try:
        import math_verify  # noqa: PLC0415
    except ImportError as error:
        _MV["module"] = None
        _MV["error"] = error
        return None, error
    _MV["module"] = math_verify
    _MV["error"] = None
    return math_verify, None


def math_verify_status() -> dict:
    module, error = _load_math_verify()
    version = None if module is None else getattr(module, "__version__", None)
    return {
        "available": module is not None,
        "version": version,
        "version_matches_article": version == MATH_VERIFY_VERSION,
        "article_version": MATH_VERIFY_VERSION,
        "error": None if error is None else f"{type(error).__name__}: {error}",
    }


def _timeout():
    # math-verify's alarm timeouts work on the main thread only. Off-thread, mathqa
    # disables them because a failing alarm would otherwise be reported as "not equal".
    if threading.current_thread() is threading.main_thread():
        return 5
    return None


def math_verify_equal(pred: str, gold: str) -> tuple[bool | None, str | None, str | None]:
    """Return ``(correct, version, error)``.

    ``correct`` is None when math-verify cannot be called with the article's
    signature. A False from verify, or an exception inside verify, matches
    mathqa.is_correct and is a real grade, not a missing grader.
    """

    module, import_error = _load_math_verify()
    if module is None:
        return None, None, f"math-verify is not installed ({import_error})"
    version = getattr(module, "__version__", None)
    timeout = _timeout()
    try:
        parsed_gold = module.parse("\\boxed{" + gold + "}", parsing_timeout=timeout)
        parsed_pred = module.parse("\\boxed{" + pred + "}", parsing_timeout=timeout)
    except TypeError as error:
        return None, version, f"math-verify parse signature mismatch: {error}"
    if not parsed_gold or not parsed_pred:
        return False, version, None
    try:
        return bool(module.verify(parsed_gold, parsed_pred, timeout_seconds=timeout)), version, None
    except TypeError as error:
        return None, version, f"math-verify verify signature mismatch: {error}"
    except Exception as error:  # noqa: BLE001
        return False, version, f"{type(error).__name__}: {error}"


def lenient_pred(text: str) -> str | None:
    """Gemma secondary reader from evaluate.lenient_pred. Not the Qwen headline."""

    found = []
    patterns = (
        r"[Ff]inal [Aa]nswer\s*(?:is)?\s*[:=]?\s*\$*([^\n$]+)",
        r"[Tt]he (?:final )?answer is\s*[:=]?\s*\$*([^\n$]+)",
        r"\b[Aa]nswer\s*[:=]\s*\$*([^\n$]+)",
    )
    for pattern in patterns:
        found += [
            (match.start(1), match.group(1).strip().strip("*").rstrip(".").strip())
            for match in re.finditer(pattern, text)
        ]
    for _, snippet in sorted(found, reverse=True):
        if re.search(r"[0-9\\]", snippet):
            return snippet
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not numbers:
        return None
    return numbers[-1].replace(",", "")


def _lenient_correct(text: str, pred, good, gold: str):
    """evaluate.lenient_correct, only when the strict grade is a real boolean."""

    if not isinstance(good, bool):
        return None
    if good or pred is not None:
        return bool(good)
    loose = lenient_pred(text)
    if loose is None:
        return False
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", loose)
    loose_grade, _, loose_error = math_verify_equal(loose, gold)
    number_grade = None
    if numbers:
        number_grade, _, number_error = math_verify_equal(numbers[-1].replace(",", ""), gold)
        loose_error = loose_error or number_error
    if loose_grade is None and number_grade is None:
        return None
    return bool(loose_grade) or bool(number_grade)


def hit_token_limit(finish_reason: str | None) -> bool | None:
    """Blog cut bit: True only when the engine reports ``length``. Unknown stays null."""

    if finish_reason is None:
        return None
    return finish_reason == "length"


def grade_response(text: str, gold: str, *, finish_reason: str | None, n_tokens: int | None = None) -> dict:
    """Grade one generation. ``correct`` follows mathqa.is_correct, or is null."""

    if not isinstance(text, str):
        raise ValueError("generation text must be a string")
    if not isinstance(gold, str) or gold == "":
        raise ValueError("gold answer must be a non-empty string")
    if n_tokens is not None and (isinstance(n_tokens, bool) or not isinstance(n_tokens, int) or n_tokens < 0):
        raise ValueError("n_tokens must be a non-negative int or None")
    pred, n_boxes = extract_answer(text)
    equiv = bool(pred is not None and is_equiv(pred, gold))
    grader_error = None
    version = None
    if pred is None:
        correct = False
        grader = "not_applicable"
    else:
        correct, version, grader_error = math_verify_equal(pred, gold)
        grader = "math-verify" if correct is not None else "unavailable"
    cut = hit_token_limit(finish_reason)
    if finish_reason is None or pred is None and cut is None:
        no_answer_finished = None
    else:
        no_answer_finished = pred is None and finish_reason != "length"
    return {
        "pred": pred,
        "n_boxes": n_boxes,
        "boxed": pred is not None,
        "correct": correct,
        "equiv": equiv,
        "equiv_checker": "hendrycks_is_equiv",
        "grader": grader,
        "grader_version": version,
        "grader_error": grader_error,
        "cut": cut,
        "no_answer_finished": no_answer_finished,
        "loop": loop_signals(answer_text(text)),
        "lenient_pred": lenient_pred(text),
        "lenient_correct": _lenient_correct(text, pred, correct, gold),
        "n_tokens": n_tokens,
        "finish_reason": finish_reason,
    }
