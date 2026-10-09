# PatchPilot eval: bench, lite20 tasks

## Component ablation

| System | Success | Tokens / task | Avg retries | Cost / task | Graded |
|---|---|---|---|---|---|
| Baseline (executor + tests) | 50% | 307.7K | 0.2 | $0.544 | 8/8 |
| + RAG | 38% | 778.0K | 0.0 | $0.714 | 8/8 |
| + Planner | 50% | 1,316.5K | 0.5 | $1.421 | 8/8 |
| + Evaluator (full PatchPilot) | 33% | 1,379.6K | 0.5 | $1.346 | 6/6 |

## All metrics

| Metric | baseline | full | planner | rag |
|---|---|---|---|---|
| Tasks (graded / attempted) | 8/8 | 6/6 | 8/8 | 8/8 |
| Task success rate | 50% | 33% | 50% | 38% |
| Tests passed (target tests) | 62% | 62% | 64% | 38% |
| Patch acceptance (applies cleanly) | 100% | 67% | 88% | 100% |
| Self-accepted by the agent | 88% | 33% | 75% | 100% |
| Self-acceptance precision | 43% | 0% | 50% | 38% |
| Invalid tool calls / task | 0.00 | 1.67 | 1.12 | 0.50 |
| Average retries | 0.25 | 0.50 | 0.50 | 0.00 |
| Input tokens / task | 297.5K | 1,347.4K | 1,285.1K | 765.7K |
| Output tokens / task | 10.2K | 32.2K | 31.4K | 12.4K |
| Tokens / task | 307.7K | 1,379.6K | 1,316.5K | 778.0K |
| Tokens / successful task | 615.5K | 4,138.7K | 2,632.9K | 2,074.7K |
| Cost / task | $0.544 | $1.346 | $1.421 | $0.714 |
| Input served from prompt cache | 29% | 70% | 65% | 69% |
| P50 latency | 98s | 494s | 428s | 3055s |
| P95 latency | 385s | 1150s | 1760s | 8867s |
| Retrieval precision (files) | n/a | 17% | 12% | 10% |
| Retrieval recall (files) | n/a | 83% | 62% | 50% |
| Mean prompt (context utilization) | 9.9K | 39.1K | 42.2K | 36.9K |
| Max prompt | 23.4K | 71.6K | 86.2K | 62.0K |
| Agent errors / environment errors | 0 / 0 | 3 / 0 | 1 / 0 | 0 / 0 |

Success and tests passed come from the official SWE-bench harness (custom tasks: their own test suite). Tokens per successful task is all tokens spent divided by tasks resolved. Retrieval precision compares the files retrieved for the prompt with the files the reference patch changes. Latency percentiles are nearest-rank, so they are observed values.
