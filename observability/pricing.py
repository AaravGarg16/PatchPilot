# USD per million tokens (input, output), Anthropic first-party list prices as of 2026-06.
# Used for the per-run cost estimate; Langfuse computes its own cost from the same usage.
PRICES_PER_MTOK = {
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        return None
    return (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000
