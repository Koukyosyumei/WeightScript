"""Token-budgeted native generation and task/loop metrics."""

import torch


@torch.inference_mode()
def generate(model, input_ids, max_new_tokens=32, temperature=0.0, seed=0):
    if not input_ids or max_new_tokens < 1 or temperature < 0:
        raise ValueError("invalid generation request")
    context = getattr(model.config, "max_position_embeddings", None) or getattr(
        model.config, "n_positions", None
    )
    if context and len(input_ids) + max_new_tokens > context:
        raise ValueError("generation exceeds model context")
    if any(
        type(t) is not int or not 0 <= t < model.config.vocab_size for t in input_ids
    ):
        raise ValueError("invalid token")
    rng = torch.Generator().manual_seed(seed)
    ids = torch.tensor([input_ids])
    generated = []
    calls = 0
    reason = "token_limit"
    eos = model.config.eos_token_id
    stops = set(eos if isinstance(eos, list) else [eos])
    for _ in range(max_new_tokens):
        logits = model(ids, use_cache=False).logits[0, -1]
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite generation logits")
        calls += 1
        token = (
            int(logits.argmax())
            if temperature == 0
            else int(
                torch.multinomial((logits / temperature).softmax(-1), 1, generator=rng)
            )
        )
        generated.append(token)
        ids = torch.cat((ids, torch.tensor([[token]])), dim=1)
        if token in stops:
            reason = "eos"
            break
    n = 4
    grams = [tuple(generated[i : i + n]) for i in range(max(0, len(generated) - n + 1))]
    return {
        "tokens": generated,
        "forward_calls": calls,
        "stop_reason": reason,
        "repeated_4gram_fraction": 1 - len(set(grams)) / len(grams) if grams else 0.0,
    }


def evaluate_generation(model, rows, max_new_tokens=32, temperature=0.0, seed=0):
    if not rows:
        raise ValueError("empty generation dataset")
    results = []
    for i, row in enumerate(rows):
        result = generate(
            model, row["input_ids"], max_new_tokens, temperature, seed + i
        )
        if "expected_ids" in row:
            result["exact_match"] = result["tokens"] == row["expected_ids"]
        results.append(result)
    scored = [r["exact_match"] for r in results if "exact_match" in r]
    return {
        "examples": len(rows),
        "forward_calls": sum(r["forward_calls"] for r in results),
        "exact_match": sum(scored) / len(scored) if scored else None,
        "token_limit_fraction": sum(r["stop_reason"] == "token_limit" for r in results)
        / len(results),
        "mean_repeated_4gram_fraction": sum(
            r["repeated_4gram_fraction"] for r in results
        )
        / len(results),
        "temperature": temperature,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "results": results,
    }
