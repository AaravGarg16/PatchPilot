# Evals

PatchPilot on SWE-bench Lite, graded by the official SWE-bench harness. Start small: the `smoke` suite is 15 Lite
tasks chosen to validate the harness cheaply (see `tasks/swebench_lite_smoke.json` for how they were picked).

## Workflow

```
# 1. Check environments: pulls each task's image, copies its repo out, runs the repo's test runner, warms the
#    embedding cache. No LLM calls, no cost. Look for "status": "ok" and "test_runner_ok": true.
python -m evals.run --suite smoke --dry-run --limit 3

# 2. Run the agent. Asks before spending; --max-cost stops starting new tasks past the cap.
python -m evals.run --suite smoke --config full --limit 3 --max-cost 2

# 3. Grade with the official harness (hidden tests run in the same task images), then report.
python -m evals.grade evals/results/smoke
python -m evals.report evals/results/smoke
```

Experiments, once the harness checks out:

```
python -m evals.run --suite smoke --experiment ablation   # baseline, + RAG, + planner, full (+ evaluator)
python -m evals.run --suite smoke --experiment context    # full (dynamic context) vs full-history
```

Reruns resume (finished tasks are skipped), so a stopped run can be continued with the same command.

Running on a GPU VM: see `docs/vm_runbook.md` (set `EMBEDDING_DEVICE=cuda` and install `fastembed-gpu`).

## What to expect

- SWE-bench images are x86_64 and run under emulation on Apple Silicon: slower, and often 1-3 GB each. 15 tasks can
  take 20-40 GB of Docker disk. Images are kept for grading, which reuses them; `docker image prune` reclaims them.
- The first time a repo is indexed it is embedded locally, which can take minutes for django or sympy. The embedding
  cache in Postgres makes every later run of that repo (other tasks, other configs) fast. The dry run warms it.
- `smoke` is biased toward small fixes, so its success rate will be higher than full Lite's. It is for validating the
  harness and comparing configs, not for quoting as a benchmark number.

## How a SWE-bench task runs

The agent works on a copy of the task image's `/testbed`, mounted read-only into a sandbox started from the same
image. Tests run with the repo's own runner (pytest, django's `runtests.py`, sympy's `bin/test`) on test files
related to the files involved, since full suites are too slow. The agent never sees the hidden tests or the reference
patch; the reference patch's file list is used only afterwards, for retrieval precision.

## Metrics

Defined in `metrics.py`; `report.md` lists them per config. Success and tests passed come from the official harness.
Tokens per successful task is all tokens spent divided by tasks resolved. Environment failures are excluded from the
rates and reported separately; agent crashes count as failures, with whatever they cost.
