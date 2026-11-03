"""Validated, executable semantic rules; no code evaluation or example lookup.

Indices address the input after explicitly declared ignored tokens are removed.
Final-token rules describe one decision. ``span_next`` describes the next token
of an explicit copy, sort, or majority program. Neither kind is evidence that
the same computation is implemented by the weights.
"""

import math


def validate_rule(rule, depth=0):
    if depth > 16 or not isinstance(rule, dict):
        raise ValueError("invalid or excessively nested semantic rule")
    op = rule.get("op")
    fields = {
        "count_threshold": {"threshold", "invert"},
        "token_at": {"index", "default"},
        "token_compare": {"left", "right", "comparison", "if_true", "if_false"},
        "count_token": {"token", "threshold", "comparison", "if_true", "if_false"},
        "select_token": {"token", "comparison", "which", "default"},
        "span_next": {"kind", "bos", "sep", "eos"},
        "branch": {"condition", "then", "else"},
    }
    if op not in fields or set(rule) - {"op", "ignore_tokens"} != fields[op]:
        raise ValueError("unknown operation or invalid semantic rule fields")
    ignored = rule.get("ignore_tokens", [])
    if not isinstance(ignored, list) or any(
        type(t) is not int or t < 0 for t in ignored
    ):
        raise ValueError("ignore_tokens must be nonnegative integer token IDs")
    for key in ("default", "token", "if_true", "if_false"):
        if key in rule and (type(rule[key]) is not int or rule[key] < 0):
            raise ValueError(f"{key} must be a nonnegative integer token ID")
    for key in ("index", "left", "right"):
        if key in rule and type(rule[key]) is not int:
            raise ValueError(f"{key} must be an integer index")
    if "comparison" in rule and rule["comparison"] not in ("eq", "lt", "gt"):
        raise ValueError("comparison must be eq, lt or gt")
    if op == "count_threshold":
        threshold = rule["threshold"]
        if (
            type(rule["invert"]) is not bool
            or type(threshold) not in (int, float)
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError("invalid count threshold")
    if op == "count_token" and (
        type(rule["threshold"]) is not int or rule["threshold"] < 0
    ):
        raise ValueError("count threshold must be a nonnegative integer")
    if op == "select_token" and rule["which"] not in ("first", "last", "min", "max"):
        raise ValueError("selection must be first, last, min or max")
    if op == "span_next":
        markers = (rule["bos"], rule["sep"], rule["eos"])
        if rule["kind"] not in ("copy", "sorted", "majority"):
            raise ValueError("span kind must be copy, sorted or majority")
        if any(type(v) is not int or v < 0 for v in markers) or len(set(markers)) < 3:
            raise ValueError("span markers must be distinct nonnegative integers")
    if op == "branch":
        for key in ("condition", "then", "else"):
            validate_rule(rule[key], depth + 1)
    return rule


def _compare(a, b, comparison):
    return {"eq": a == b, "lt": a < b, "gt": a > b}[comparison]


def rule_predict(rule, ids):
    """Execute a validated rule, raising on undefined index/empty binary domains."""
    validate_rule(rule)
    if not ids or any(type(t) is not int or t < 0 for t in ids):
        raise ValueError("rule input must be nonempty nonnegative integer token IDs")
    tokens = [t for t in ids if t not in rule.get("ignore_tokens", [])]
    op = rule["op"]
    if op == "count_threshold":
        bits = [t for t in tokens if t in (0, 1)]
        if not bits:
            raise ValueError("rule domain needs at least one bit")
        return int((sum(bits) / len(bits) > rule["threshold"]) != rule["invert"])
    if op == "token_at":
        index = rule["index"]
        return tokens[index] if -len(tokens) <= index < len(tokens) else rule["default"]
    if op == "token_compare":
        try:
            yes = _compare(
                tokens[rule["left"]], tokens[rule["right"]], rule["comparison"]
            )
        except IndexError as exc:
            raise ValueError("token comparison index outside rule domain") from exc
        return rule["if_true"] if yes else rule["if_false"]
    if op == "count_token":
        yes = _compare(
            tokens.count(rule["token"]), rule["threshold"], rule["comparison"]
        )
        return rule["if_true"] if yes else rule["if_false"]
    if op == "select_token":
        selected = [t for t in tokens if _compare(t, rule["token"], rule["comparison"])]
        if not selected:
            return rule["default"]
        if rule["which"] in ("min", "max"):
            return (min if rule["which"] == "min" else max)(selected)
        return selected[0 if rule["which"] == "first" else -1]
    if op == "span_next":
        return _span_next(rule, tokens)
    # A condition token of 0 takes the else branch. Every other token id is true.
    condition = rule_predict(rule["condition"], tokens)
    return rule_predict(rule["then"] if condition else rule["else"], tokens)


def _span_next(rule, tokens):
    """Next token of a gold-aligned copy, ascending sort, or binary majority span."""
    bos, sep, eos = rule["bos"], rule["sep"], rule["eos"]
    if tokens.count(bos) != 1 or tokens.count(sep) != 1:
        raise ValueError("span rule domain needs one bos and one sep")
    begin, split = tokens.index(bos), tokens.index(sep)
    if begin >= split or eos in tokens[begin : split + 1]:
        raise ValueError("bos must precede sep with no eos in the source span")
    source, generated = tokens[begin + 1 : split], tokens[split + 1 :]
    if rule["kind"] == "copy":
        target = source + [eos]
    elif rule["kind"] == "sorted":
        target = sorted(source) + [eos]
    else:
        bits = [token for token in source if token in (0, 1)]
        if not bits or bits.count(0) == bits.count(1):
            raise ValueError("majority span is undefined")
        target = [0 if bits.count(0) > bits.count(1) else 1]
    if generated != target[: len(generated)] or len(generated) >= len(target):
        raise ValueError("span prefix is outside the rule program")
    return target[len(generated)]


def supervised_pairs(ids, sep):
    """Pairs ``(prefix, gold next token)`` for every position after the one sep."""
    if (
        not ids
        or any(type(token) is not int or token < 0 for token in ids)
        or type(sep) is not int
    ):
        raise ValueError("supervised span input is invalid")
    if ids.count(sep) != 1:
        raise ValueError("supervised span requires one sep")
    start = ids.index(sep)
    if start == len(ids) - 1:
        raise ValueError("supervised span has no target token")
    return [(ids[:index], ids[index]) for index in range(start + 1, len(ids))]
