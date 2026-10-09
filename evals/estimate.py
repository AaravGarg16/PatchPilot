"""Estimate what the benchmark plan costs before spending anything. No LLM calls.

    python -m evals.estimate                                  # assumed tokens per run (see ASSUMED)
    python -m evals.estimate --measured evals/results/bench   # averages from runs already done (any suite)

Counts each (config, task) pair once: lite12 contains every ablation8 task, so `full` on those runs once and serves
the ablation, the headline and the context experiment.
"""
import argparse
from pathlib import Path

from evals import results as store
from evals.grade import config_dirs
from evals.tasks import suite_instance_ids
from observability.pricing import cost_usd

# The benchmark: which configs run on which suite. Shared pairs are counted once.
PLAN = [
    ("Component ablation", "ablation8", ["baseline", "rag", "planner", "full"]),
    ("Headline (full agent)", "lite12", ["full"]),
    ("Context engineering", "ablation8", ["full", "full-history"]),
]

# Tokens per agent run (input, output) when nothing has been measured yet. Derived from the agent's own limits, not
# from data: a 40k-token context budget, executor prompts of ~20-30k tokens, ~10-15 executor turns per attempt,
# ~1.2 attempts when tests alone decide (they usually pass on the first try) and ~1.6 with the evaluator. Full history
# keeps every tool output and carries the transcript into retries, so assume about double. Replace with --measured.
ASSUMED = {
    "baseline": (250_000, 5_000),
    "rag": (280_000, 5_000),
    "planner": (290_000, 6_000),
    "full": (450_000, 9_000),
    "full-history": (900_000, 9_000),
}


# Share of each run's input tokens (read from cache, written to cache) with prompt caching on. Assumed from the
# prompt shapes, not measured:
# - Executor turns re-send the stable prefix (tools, system, task message), cached for the whole attempt. With
#   retrieval it is ~20K of a ~28K prompt; the baseline has no retrieved code, so its prefix is only ~4K.
# - In dynamic mode the tail is cached only until older tool outputs start being trimmed; full history only appends,
#   so nearly all of each prompt is a read and only the newest messages are written.
# - Planner and evaluator calls (~10% of input in configs that have them) are not cached.
ASSUMED_CACHE = {
    "baseline": (0.27, 0.03),
    "rag": (0.60, 0.08),
    "planner": (0.62, 0.08),
    "full": (0.58, 0.07),
    "full-history": (0.80, 0.08),
}


def measured_averages(run_dir: Path) -> dict[str, tuple]:
    """config -> (mean input, mean output, runs, read share, write share), from finished non-env-error runs."""
    averages = {}
    for d in config_dirs(run_dir):
        runs = [r for r in store.load_all(d) if r.status != "env_error" and r.llm_calls]
        if runs:
            total_in = sum(r.input_tokens for r in runs) or 1
            cached = any(r.cache_read_tokens or r.cache_write_tokens for r in runs)
            shares = ((sum(r.cache_read_tokens for r in runs) / total_in, sum(r.cache_write_tokens for r in runs) / total_in)
                      if cached else None)
            averages[d.name] = (sum(r.input_tokens for r in runs) // len(runs),
                                sum(r.output_tokens for r in runs) // len(runs), len(runs), shares)
    return averages


def run_cost(model: str, config: str, tokens_in: int, tokens_out: int, cache: tuple[float, float] | None) -> float:
    if cache is None:
        return cost_usd(model, tokens_in, tokens_out) or 0.0
    read, write = cache
    return cost_usd(model, round(tokens_in * (1 - read - write)), tokens_out,
                    cache_read_tokens=round(tokens_in * read), cache_write_tokens=round(tokens_in * write)) or 0.0


def estimate(model: str, measured: dict | None = None, done: set[tuple[str, str]] | None = None) -> list[dict]:
    measured = measured or {}
    done = done or set()
    counted: set[tuple[str, str]] = set()
    rows = []
    for name, suite, configs in PLAN:
        pairs = [(c, i) for c in configs for i in suite_instance_ids(suite)]
        new = [p for p in pairs if p not in counted and p not in done]
        counted.update(pairs)
        cost = {"standard": 0.0, "cached": 0.0}
        for config, _ in new:
            tokens_in, tokens_out = measured.get(config, ASSUMED[config])[:2]
            m = measured.get(config)
            cache = m[3] if m and len(m) > 3 and m[3] is not None else ASSUMED_CACHE[config]
            cost["standard"] += run_cost(model, config, tokens_in, tokens_out, None)
            cost["cached"] += run_cost(model, config, tokens_in, tokens_out, cache)
        rows.append({"experiment": name, "suite": suite, "runs": len(pairs), "new_runs": len(new), **cost})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--measured", type=Path, help="results folder to take per-config token averages from")
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--budget", type=float, default=15.0)
    args = parser.parse_args()

    measured = measured_averages(args.measured) if args.measured else {}
    done = set()
    if args.measured:
        done = {(d.name, r.instance_id) for d in config_dirs(args.measured) for r in store.load_all(d)}
    print(f"Per run ({'measured where available' if measured else 'assumed; pass --measured to use real runs'}):")
    for config, (i, o) in ASSUMED.items():
        m = measured.get(config)
        mi, mo = (m[0], m[1]) if m else (i, o)
        read, write = m[3] if m and m[3] is not None else ASSUMED_CACHE[config]
        source = f"measured over {m[2]} run(s)" if m else "assumed"
        print(f"  {config:13} {mi / 1000:5.0f}K in  {mo / 1000:4.1f}K out  cache {read:4.0%} read / {write:3.0%} write  ({source})")
    rows = estimate(args.model, measured, done)
    print(f"\n{'Experiment':24} {'Suite':10} {'Runs':>5} {'New':>5} {'No cache':>10} {'With cache':>11}")
    for r in rows:
        print(f"{r['experiment']:24} {r['suite']:10} {r['runs']:5} {r['new_runs']:5} ${r['standard']:9.2f} ${r['cached']:10.2f}")
    total_std, total_cached = sum(r["standard"] for r in rows), sum(r["cached"] for r in rows)
    print(f"{'Total (new runs only)':41} ${total_std:9.2f} ${total_cached:10.2f}")
    verdict = "fits" if total_cached <= args.budget else f"is ${total_cached - args.budget:.2f} over"
    print(f"\nStandard API with prompt caching: the plan {verdict} a ${args.budget:.2f} budget.")


if __name__ == "__main__":
    main()
