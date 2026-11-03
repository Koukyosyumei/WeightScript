"""Repetition signals from the published math harness.

``verbatim_loop_end`` is the detector in exp8_math/code/mathqa.py ``loop_signals``.
``repetitive_end`` is the rule that function's docstring gives for
``mcqa.loop_signals``. That helper is not in the public exp8 release, so the
boolean here follows the stated rule and is not a byte-identical port of the
unpublished function.

The blog table's looping column is not either of these. It is the share of
answers whose generation finish reason is ``length`` (the 32,768-token cap).
"""

from __future__ import annotations


LOOP_MAX_PERIOD = 200
LOOP_MIN_REPEATS = 3
LOOP_MIN_SPAN = 40
REPETITIVE_WINDOW = 200
REPETITIVE_NGRAM = 4
REPETITIVE_MIN_COUNT = 3


def verbatim_loop_end(words: list[str]) -> bool:
    """True when the end, aside from its last word, is one block repeated in a row.

    The block is 1..200 words, repeated at least 3 times and spanning at least
    40 words. The last word is dropped first: a token-capped reply usually
    stops inside a word.
    """

    trimmed = words[:-1]
    n = len(trimmed)
    for period in range(1, LOOP_MAX_PERIOD + 1):
        repeats = max(LOOP_MIN_REPEATS, -(-LOOP_MIN_SPAN // period))
        if repeats * period > n:
            break
        block = trimmed[n - period :]
        if all(
            trimmed[n - (j + 1) * period : n - j * period] == block
            for j in range(1, repeats)
        ):
            return True
    return False


def repetitive_end(words: list[str]) -> bool:
    """True when, in the last 200 words, at least half of the 4-word positions
    are a phrase that occurs at least 3 times in that window.

    This is the rule stated in ``mathqa.loop_signals``. The window is the full
    word list, including the last word. ``verbatim_loop_end`` drops that word
    on its own.
    """

    window = words[-REPETITIVE_WINDOW:]
    positions = len(window) - (REPETITIVE_NGRAM - 1)
    if positions <= 0:
        return False
    counts: dict[tuple[str, ...], int] = {}
    grams: list[tuple[str, ...]] = []
    for index in range(positions):
        gram = tuple(window[index : index + REPETITIVE_NGRAM])
        grams.append(gram)
        counts[gram] = counts.get(gram, 0) + 1
    hot = sum(1 for gram in grams if counts[gram] >= REPETITIVE_MIN_COUNT)
    return hot * 2 >= positions


def loop_signals(text: str) -> dict:
    words = text.split()
    return {
        "verbatim_loop_end": verbatim_loop_end(words),
        "repetitive_end": repetitive_end(words),
    }
