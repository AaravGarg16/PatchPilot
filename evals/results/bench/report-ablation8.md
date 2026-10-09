# PatchPilot eval: bench, ablation8 tasks

## Component ablation

| System | Success | Tokens / task | Avg retries | Cost / task | Graded |
|---|---|---|---|---|---|
| Baseline (executor + tests) | 50% | 307.7K | 0.2 | $0.544 | 8/8 |
| + RAG | 38% | 778.0K | 0.0 | $0.714 | 8/8 |
| + Planner | 60% | 1,663.1K | 0.8 | $1.903 | 5/5 |

## All metrics

| Metric | baseline | planner | rag |
|---|---|---|---|
| Tasks (graded / attempted) | 8/8 | 5/5 | 8/8 |
| Task success rate | 50% | 60% | 38% |
| Tests passed (target tests) | 62% | 60% | 38% |
| Patch acceptance (applies cleanly) | 100% | 100% | 100% |
| Self-accepted by the agent | 88% | 80% | 100% |
| Self-acceptance precision | 43% | 50% | 38% |
| Invalid tool calls / task | 0.00 | 1.00 | 0.50 |
| Average retries | 0.25 | 0.80 | 0.00 |
| Input tokens / task | 297.5K | 1,621.9K | 765.7K |
| Output tokens / task | 10.2K | 41.1K | 12.4K |
| Tokens / task | 307.7K | 1,663.1K | 778.0K |
| Tokens / successful task | 615.5K | 2,771.8K | 2,074.7K |
| Cost / task | $0.544 | $1.903 | $0.714 |
| Input served from prompt cache | 29% | 61% | 69% |
| P50 latency | 98s | 418s | 3055s |
| P95 latency | 385s | 1760s | 8867s |
| Retrieval precision (files) | n/a | 16% | 10% |
| Retrieval recall (files) | n/a | 80% | 50% |
| Mean prompt (context utilization) | 9.9K | 41.8K | 36.9K |
| Max prompt | 23.4K | 86.2K | 62.0K |
| Agent errors / environment errors | 0 / 0 | 0 / 0 | 0 / 0 |

Success and tests passed come from the official SWE-bench harness (custom tasks: their own test suite). Tokens per successful task is all tokens spent divided by tasks resolved. Retrieval precision compares the files retrieved for the prompt with the files the reference patch changes. Latency percentiles are nearest-rank, so they are observed values.
