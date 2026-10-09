# USD per million tokens (input, output), Anthropic first-party list prices as of 2026-06.
# Used for the per-run cost estimate; Langfuse computes its own cost from the same usage.
PRICES_PER_MTOK = {
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


# The Message Batches API bills input and output at half the standard rate.
BATCH_DISCOUNT = 0.5
# Prompt caching, relative to the base input price: writing an entry costs extra (more for the longer TTL), reading
# one costs a tenth.
CACHE_WRITE_5M = 1.25
CACHE_WRITE_1H = 2.0
CACHE_READ = 0.1


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    batch: bool = False,
    *,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    cache_write_1h_tokens: int = 0,
) -> float | None:
    """input_tokens is the uncached input only; cache reads and writes are priced separately."""
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        return None
    effective_input = (input_tokens + cache_read_tokens * CACHE_READ + cache_write_tokens * CACHE_WRITE_5M
                       + cache_write_1h_tokens * CACHE_WRITE_1H)
    cost = (effective_input * prices[0] + output_tokens * prices[1]) / 1_000_000
    return cost * BATCH_DISCOUNT if batch else cost
