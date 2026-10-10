# PatchPilot eval: bench2, ablation8 tasks

## All metrics

| Metric | full |
|---|---|
| Tasks (graded / attempted) | 5/5 |
| Task success rate | 0% |
| Tests passed (target tests) | 0% |
| Patch acceptance (applies cleanly) | 60% |
| Self-accepted by the agent | 40% |
| Self-acceptance precision | 0% |
| Invalid tool calls / task | 0.80 |
| Average retries | 0.40 |
| Input tokens / task | 1,091.6K |
| Output tokens / task | 29.0K |
| Tokens / task | 1,120.6K |
| Tokens / successful task | n/a |
| Cost / task | $1.231 |
| Input served from prompt cache | 65% |
| P50 latency | 574s |
| P95 latency | 957s |
| Retrieval precision (files) | 16% |
| Retrieval recall (files) | 80% |
| Mean prompt (context utilization) | 37.4K |
| Max prompt | 71.7K |
| Agent errors / environment errors | 3 / 0 |
| Runs that used a fallback step | 0 |

Success and tests passed come from the official SWE-bench harness (custom tasks: their own test suite). Tokens per successful task is all tokens spent divided by tasks resolved. Retrieval precision compares the files retrieved for the prompt with the files the reference patch changes. Latency percentiles are nearest-rank, so they are observed values.
