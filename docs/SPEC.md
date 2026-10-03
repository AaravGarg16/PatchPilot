# PatchPilot — Full Spec

Deep reference for the project. CLAUDE.md holds the operating rules and phase order; this file holds the detail each phase needs. Nothing here is a license to skip the phase discipline in CLAUDE.md.

## Core idea

Give PatchPilot a public GitHub repository and an issue. It understands the codebase, retrieves relevant files, plans a solution, edits the code, runs tests inside an isolated sandbox, evaluates its own patch, retries when necessary, and returns a validated diff.

## Workflow

GitHub repo + issue, then: repository ingestion, code-aware RAG / vector search, planner agent, coding agent, MCP / tool calls, Docker sandbox, run tests + lint, evaluator / judge. If it passes, return the final diff, explanation, and metrics. If it fails, diagnose and retry.

## Stack

- Frontend: Next.js + TypeScript + React
- Backend: FastAPI + Python
- Orchestration: LangGraph
- LLM utilities: LangChain
- RAG: embeddings + pgvector
- Database: PostgreSQL / Supabase
- Tool protocol: MCP
- Execution: Docker
- Observability: Langfuse
- CI/CD: GitHub Actions
- Version control: Git / GitHub API

PostgreSQL + pgvector rather than a separate vector database at first. Known tech, gives the vector-search keyword legitimately, keeps the architecture manageable.

## What the agent should do

Given a repo URL and an issue, PatchPilot autonomously:

1. Clone the repository into an isolated workspace.
2. Inspect the repository structure and identify language and framework.
3. Parse and index the codebase.
4. Store code embeddings in pgvector.
5. Retrieve only code relevant to the issue.
6. Generate a structured implementation plan.
7. Select files that need modification.
8. Modify those files through controlled tools.
9. Run formatting, linting, and existing tests inside Docker.
10. Run newly generated tests where appropriate.
11. Analyze failures.
12. Have an evaluator agent judge the patch.
13. Retry with failure feedback if needed.
14. Stop after a configurable retry budget.
15. Return the final diff, test results, summary, token usage, latency, cost, and success/failure status.

## Agent architecture

Three roles. Do not add more just to say "multi-agent."

### Planner

Takes the issue plus retrieved code and produces a structured plan:

```json
{
  "problem": "...",
  "relevant_files": [],
  "implementation_steps": [],
  "tests_needed": [],
  "risks": []
}
```

### Executor

Uses tools: read_file, search_code, list_directory, edit_file, create_file, run_tests, run_linter, git_diff. The executor does not receive the entire repository. It gets dynamically selected context.

### Evaluator

Receives the original issue, implementation plan, git diff, test output, lint output, and previous failures. Returns:

```json
{
  "correct": true,
  "requirements_met": true,
  "tests_pass": true,
  "problems": [],
  "recommended_action": "finish"
}
```

If it fails, LangGraph routes the workflow back to the executor.

This gives legitimate: multi-agent systems, agent orchestration, agent evaluation, feedback loops, tool calling, state management.

## LangGraph design

Graph: START, analyze_repository, retrieve_context, plan, implement, run_tests, evaluate, then a success branch. On success, finalize. On failure, diagnose and route back to implement.

State fields:

- repo_id, issue, repo_metadata
- retrieved_files, context_summary
- plan
- modified_files, diff
- test_results, lint_results
- evaluation
- attempt_number, max_attempts
- token_usage, cost, latency

## RAG system

Not basic PDF-style RAG. Code-aware. Instead of fixed-size character splits, chunk around functions, classes, methods, modules, configuration blocks, and README / documentation sections.

Store metadata such as:

```json
{
  "repository": "...",
  "path": "app/auth/service.py",
  "language": "python",
  "symbol": "authenticate_user",
  "type": "function"
}
```

Retrieval pipeline: issue, query rewriting, vector search, keyword / symbol search, metadata filtering, reranking, top relevant code chunks.

Covers: RAG, embeddings, vector databases, semantic search, retrieval, reranking, code-aware chunking.

## Context engineering

Do not append every message and tool response to the prompt. Use a token budget:

- System instructions: 10%
- Task / issue: 10%
- Current plan: 10%
- Retrieved source code: 45%
- Previous failure summary: 15%
- Recent tool outputs: 10%

Techniques: dynamic context selection, history summarization, old-tool-output compression, diff-aware context, retrieval refresh after file edits, token-budget enforcement.

Track: input tokens / task, output tokens / task, tokens / successful task, context utilization.

Experiment: compare a baseline that uses full conversation history against PatchPilot's dynamic context engineering. Target metric form: reduced context consumption by X% while maintaining equivalent task success. This is a separate experiment from the component ablation and is one of the strongest resume lines.

## MCP

Use MCP for real functionality. Good MCP-exposed capabilities: filesystem, Git operations, repository search, GitHub issue retrieval, test execution interface, documentation lookup. The agent discovers available tools from an MCP server rather than hardcoding every integration.

Interview framing: MCP standardized how the agent discovered and interacted with external tools, while LangGraph managed the agent workflow itself.

## Docker sandbox

Never execute arbitrary repository code directly on the host. Each run: create a temporary container, clone the repo, install dependencies, let the agent modify files, execute tests, collect outputs, destroy the container.

Limits: CPU limit, memory limit, execution timeout, disk limit, non-root user, restricted filesystem, network disabled after dependency installation where possible.

For the public demo, do not let strangers execute arbitrary repositories. Use 5 to 10 curated repos and issues first.

## Observability

Langfuse on every run. Trace: LLM calls, tool calls, retrieval results, planning, code modifications, test execution, evaluation, retry loops.

Store per run: model, task, success, tokens, latency, cost, tool failures, number of retries, tests before, tests after.

The UI should let someone inspect the execution step by step: planning, retrieved files, modifications, test runs, evaluator findings, retries, final result.

## Evaluation

The part that makes the project stand out. Spend serious effort here.

Primary eval set: SWE-bench Lite (chosen over hand-authoring tasks; recognizable numbers, saves weeks). Supplement later with custom tasks toward 100 to 150.

Custom task types: bug fixes, small feature additions, test generation, API changes, refactoring, configuration fixes, dependency problems.

Metrics: task success rate, tests passed, patch acceptance rate, invalid tool calls, average retries, tokens / task, tokens / successful task, cost / task, P50 latency, P95 latency, retrieval precision.

Component ablation: baseline agent, + RAG, + planner, + evaluator, full PatchPilot. Produce a results table (success, tokens, avg retries). Example shape only, real numbers from real runs:

| System      | Success | Tokens | Avg retries |
|-------------|---------|--------|-------------|
| Baseline    | 41%     | 31K    | 2.8         |
| + RAG       | 55%     | 22K    | 2.2         |
| + Planner   | 63%     | 21K    | 1.8         |
| + Evaluator | 72%     | 23K    | 1.4         |

This experimental component is what separates "I built a LangChain agent" from "I engineered and evaluated an autonomous AI system."

## Frontend

Polished interface. Main page: pick a curated demo repo, enter or select an issue, run the agent.

Live workflow cards during a run: repository analyzed, code chunks indexed, relevant files retrieved, plan created, editing code, running tests, evaluating patch.

Results page: status (resolved / not), tests passed, files changed and line counts, agent attempts, tokens, latency, estimated cost. Tabs: plan, diff, tests, agent trace, retrieved context, metrics.

## Demo strategy

Two modes.

Instant demo: replay previously completed traces. A recruiter clicks one and immediately sees issue, reasoning summary, edits, failed tests, recovery, successful patch. No API cost, no waiting.

Live demo: a small number of controlled real executions on curated repos. Paste-any-public-repo comes only after sandbox security is solid.

## CI/CD

GitHub Actions. On every PR: Python tests, TypeScript tests, lint, type checking, agent unit tests, small eval suite, Docker image build. On main: build, full evaluation suite, deploy backend, deploy frontend.

## Repository structure

```
patchpilot/
  apps/
    web/                 # Next.js
    api/                 # FastAPI
  agent/
    graph.py             # LangGraph
    state.py
    planner.py
    executor.py
    evaluator.py
    prompts/
  retrieval/
    parser.py
    chunker.py
    embeddings.py
    retriever.py
    reranker.py
  tools/
    filesystem/
    git/
    testing/
    mcp/
  sandbox/
    Dockerfile
    runner.py
    security.py
  evals/
    tasks/
    evaluator.py
    metrics.py
  observability/
    langfuse.py
  tests/
  docs/
    SPEC.md
  .github/
    workflows/
```

## Build order

Mapped onto the phases in CLAUDE.md. The phase numbering there is authoritative; this is the original intent behind it.

- Week 1, make the agent work: repo clone, code search, planner, editor tools, test runner, diff, basic LangGraph workflow.
- Week 2, make it intelligent: RAG, pgvector, code-aware chunking, evaluator agent, retry loop, context management, MCP.
- Week 3, make it production-like: Docker sandbox, Langfuse, eval harness, metrics, CI/CD, FastAPI, Next.js UI.
- Week 4, make it impressive: benchmark experiments, token optimization, latency optimization, live demo, architecture diagram, README, demo video, resume metrics.

The weeks are the shape of the work, not a literal timeline. With an internship and classes it takes longer, and that is fine.

## What not to waste time on

Do not: train your own LLM, add TensorFlow for a keyword, build ten agents, support every programming language, build GitHub OAuth immediately, allow unrestricted shell access publicly, or spend two weeks on a beautiful UI before the agent works. The intelligence and evaluation system matter more.

## Future work (post-ship)

Not part of the 8 phases. README "future work" lines, and only to build after the project ships and the evals are solid:

- GEPA optimization: use the eval signal (tests passing, evaluator score) to evolve the agent's prompts with GEPA. Sample-efficient, modern, gives a clean before/after task-success metric without training compute.
- Rejection-sampling fine-tune (optional): keep only passing trajectories, fine-tune a small model on them, measure before vs after. A real weight-training story if that angle is wanted.

Heavy online RL (GRPO/PPO on the agent policy) is deliberately out of scope. Too many rollouts, too much compute, high risk of not finishing.

## Resume-ready criteria

Put it on the resume once there is: a working agent, RAG, LangGraph, tool calling, Docker sandbox, tests, evaluator / retry loop, a public demo, at least ~30 benchmark tasks, and real performance metrics. Keep improving after applying.

## Target resume entry

Numbers go in only after they are measured.

PatchPilot | Autonomous Software Engineering Agent
Python, LangGraph, LangChain, MCP, RAG, FastAPI, Next.js, Docker, PostgreSQL/pgvector

Built and deployed a multi-agent software engineering system using LangGraph, MCP, and code-aware RAG to autonomously analyze GitHub issues, modify repositories, execute tests in Docker sandboxes, and iteratively repair failed patches. Developed an agent evaluation and observability pipeline across 100+ coding tasks using Langfuse, improving task success from X% to Y% while reducing context tokens by Z% through dynamic retrieval and context engineering.
