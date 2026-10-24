"""Token counts and cost for the agents' model calls."""

# $ per million tokens: input, cached input, output (gpt-6-luna, MGT 409 lecture 11)
PRICES = {"gpt-6-luna": (0.10, 0.01, 0.50)}


def new_trace(model, **extra):
    return {"model": model, "model_calls": 0, "tokens_in": 0, "tokens_cached": 0,
            "tokens_out": 0, "tokens_reasoning": 0, "cost_usd": 0.0, **extra}


def add(trace, usage):
    """Count one response's usage into the trace. Input tokens include the
    cached ones, which are billed at the cached rate."""
    cached = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0
    reasoning = getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0) or 0
    trace["model_calls"] += 1
    trace["tokens_in"] += usage.input_tokens
    trace["tokens_cached"] += cached
    trace["tokens_out"] += usage.output_tokens
    trace["tokens_reasoning"] += reasoning
    trace["cost_usd"] = cost(trace)
    return {"tokens_in": usage.input_tokens, "tokens_cached": cached,
            "tokens_out": usage.output_tokens}


def cost(trace):
    price_in, price_cached, price_out = PRICES.get(trace["model"], PRICES["gpt-6-luna"])
    fresh = trace["tokens_in"] - trace.get("tokens_cached", 0)
    return round((fresh * price_in + trace.get("tokens_cached", 0) * price_cached
                  + trace["tokens_out"] * price_out) / 1e6, 6)
