"""Fixed, seeded benchmark task sets drawn from SWE-bench Lite.

    python -m evals.task_sets     # regenerates evals/tasks/<name>.json; identical output for the same dataset and seed

How the sets are drawn, so nobody can pick easy tasks:
- Development tasks (the smoke suite, used to build and tune the harness) are excluded from every set.
- Each repo's remaining tasks are shuffled once with a seed derived from SEED and the repo name. A set that needs k
  tasks from a repo takes the first k in that order. Sets that sample the same repo therefore share tasks by
  construction, and no draw depends on which other repos a set includes.
- How many tasks each repo gets is proportional to its share of the pool the set samples from (largest remainder).
"""
import json
import math
import random
from collections import Counter
from pathlib import Path

from evals.tasks import EVALS_DIR, SWEBENCH_LITE

SEED = 20261007
DEV_SUITE = EVALS_DIR / "tasks" / "swebench_lite_smoke.json"

# name -> (size, repos to sample from or None for all of Lite, purpose)
SETS = {
    "ablation8": (8, ["django/django", "sympy/sympy"],
                  ("Component ablation and context experiment. Limited to the two largest Lite repos, whose indexes "
                   "are reused across tasks; within them, tasks are a seeded random draw.")),
    "lite12": (12, None,
               "Headline success rate. Repos in proportion to SWE-bench Lite; tasks a seeded random draw within each."),
}


def allocate(counts: dict[str, int], size: int) -> dict[str, int]:
    """Largest-remainder apportionment of `size` slots across repos, proportional to their task counts."""
    total = sum(counts.values())
    quotas = {repo: size * n / total for repo, n in counts.items()}
    alloc = {repo: math.floor(q) for repo, q in quotas.items()}
    # Ties broken by repo name so the result never depends on dict order.
    for repo in sorted(quotas, key=lambda r: (-(quotas[r] - alloc[r]), r))[: size - sum(alloc.values())]:
        alloc[repo] += 1
    return {repo: n for repo, n in alloc.items() if n}


def repo_order(pool: list[str], repo: str, seed: int = SEED) -> list[str]:
    ordered = sorted(pool)
    random.Random(f"{seed}:{repo}").shuffle(ordered)
    return ordered


def build_sets(rows: list[dict], dev_ids: set[str], seed: int = SEED) -> dict[str, dict]:
    by_repo: dict[str, list[str]] = {}
    for row in rows:
        if row["instance_id"] not in dev_ids:
            by_repo.setdefault(row["repo"], []).append(row["instance_id"])
    orders = {repo: repo_order(ids, repo, seed) for repo, ids in by_repo.items()}
    lite_counts = Counter(row["repo"] for row in rows)  # proportions follow Lite itself, dev tasks included

    sets = {}
    for name, (size, repos, purpose) in SETS.items():
        counts = {r: n for r, n in lite_counts.items() if repos is None or r in repos}
        alloc = allocate(counts, size)
        ids = [i for repo in sorted(alloc) for i in orders[repo][: alloc[repo]]]
        sets[name] = {
            "dataset": SWEBENCH_LITE,
            "seed": seed,
            "purpose": purpose,
            "selection": __doc__.split("How the sets are drawn, so nobody can pick easy tasks:\n", 1)[1].strip(),
            "excluded_dev_tasks": sorted(dev_ids),
            "allocation": alloc,
            "instance_ids": ids,
        }
    return sets


def main():
    from datasets import load_dataset

    dataset = load_dataset(SWEBENCH_LITE, split="test")
    dev_ids = set(json.loads(DEV_SUITE.read_text())["instance_ids"])
    rows = [{"instance_id": r["instance_id"], "repo": r["repo"]} for r in dataset]
    for name, spec in build_sets(rows, dev_ids).items():
        spec["dataset_fingerprint"] = dataset._fingerprint
        path = EVALS_DIR / "tasks" / f"{name}.json"
        path.write_text(json.dumps(spec, indent=2) + "\n")
        print(f"{path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}: {spec['allocation']}")


if __name__ == "__main__":
    main()
