# PatchPilot

Autonomous software engineering agent. Takes a GitHub repo plus an issue, understands the code, plans a fix, edits files, runs tests in an isolated sandbox, evaluates its own patch, retries on failure, and returns a validated diff.

This is a portfolio project meant to demonstrate production AI engineering, not a chatbot.

The full spec and all deep detail (state fields, token budget, retrieval pipeline, metric list, UI layout, demo modes) live in docs/SPEC.md. This file is the operating rules and phase discipline. Read SPEC.md when a phase needs the detail.

## Prime directive

Build only the current phase. When it works, stop and wait for me. Never scaffold a later phase early. If a task tempts you toward a later phase (RAG, Docker, UI before their phase), say so and stop instead of doing it.

Every phase ends the same way: something runs, tests pass, and we commit.

## Current phase

Phase 5: Safety and observability.

## Tech stack

Add a technology only when its phase arrives. Never add one just for a keyword.

- Frontend: Next.js, TypeScript, React
- Backend: Python, FastAPI
- Orchestration: LangGraph for workflow and state. LangChain only for model wrappers, retrievers, and structured outputs.
- Tool protocol: MCP. Expose the executor's tools through an MCP server so the agent discovers them rather than hardcoding each integration. MCP handles tool discovery, LangGraph handles the workflow.
- Retrieval: PostgreSQL + pgvector, semantic embeddings, symbol-aware chunking, hybrid semantic + keyword search, optional reranking
- Execution: Docker sandbox, disposable per run
- Observability: Langfuse
- Evals: SWE-bench Lite plus a custom harness
- CI/CD: GitHub Actions
- VCS: Git, GitHub API

## Agents

Three roles, no more.

- Planner: issue plus retrieved code produces a structured plan (problem, relevant_files, implementation_steps, tests_needed, risks).
- Executor: uses tools (read_file, search_code, list_directory, edit_file, create_file, run_tests, run_linter, git_diff). Receives dynamically selected context, never the whole repo.
- Evaluator: given the issue, plan, diff, test output, lint output, and prior failures, returns structured judgment (correct, requirements_met, tests_pass, problems, recommended_action). On failure, LangGraph routes back to the executor.

## Build phases

1. Setup. Repo, venv, deps, .env, this file, one empty LangGraph node that prints and returns.
2. Core loop. One hardcoded repo and issue. Clone, plan, edit, run tests in a subprocess, diff, retry edge. Fix one real bug end to end.
3. Real agent. Full executor tool set, then expose those tools through an MCP server so the agent discovers them instead of hardcoding each. Evaluator node with structured judgment and a retry that feeds failures back in.
4. Retrieval. pgvector, symbol-aware chunking, selected-file retrieval, token budget, context engineering.
5. Safety and observability. Real Docker sandbox (non-root, CPU, memory, time limits, disposable). Langfuse tracing on every call and retry.
6. Evals. SWE-bench Lite, metrics harness, and two experiments: the component ablation (baseline, plus RAG, plus planner, plus evaluator, full system), and the context-engineering comparison (full-history baseline vs dynamic context, reporting token reduction at equal task success). Full metric list in SPEC.md.
7. Product surface. FastAPI endpoints, Next.js UI with live workflow cards, instant-replay demo mode.
8. Ship. GitHub Actions CI, README, architecture diagram, demo video, final metrics.

## Target layout

Keep modules separate. Do not collapse concerns into one place.

```
patchpilot/
  apps/
    web/            # Next.js (Phase 7)
    api/            # FastAPI (Phase 7)
  agent/
    graph.py        # LangGraph
    state.py
    planner.py
    executor.py
    evaluator.py
    prompts/
  retrieval/        # Phase 4
  tools/            # filesystem, git, testing, mcp
  sandbox/          # Phase 5
  evals/            # Phase 6
  observability/    # Phase 5
  tests/
  docs/
    SPEC.md         # full reference spec
  .github/workflows/
```

## Engineering rules

- Structured outputs everywhere. Typed Pydantic schemas for plans, evaluations, and state.
- Strict separation: orchestration, retrieval, tools, sandbox, evaluation, observability, api, frontend.
- Build the smallest robust version of a feature, then tests, then verify.
- Prefer clear and testable over clever.
- Do not over-engineer. Do not create agents beyond the three above.

## Per-feature workflow

For each feature:
1. Explain the architecture decision.
2. Explain why the tech is appropriate.
3. Name the failure modes.
4. Implement the smallest robust version.
5. Write tests.
6. Tell me how to verify manually.
7. Tell me what metric to collect.

## Security hard lines

- Never execute repository code on the host. From Phase 5 on, all execution runs in a disposable Docker container.
- Before Phase 5, only run curated repos I trust, in a subprocess, on my machine.
- No arbitrary or user-submitted repos until the sandbox is complete.
- Never expose API keys or secrets to the browser or the sandbox. Secrets live in .env only.

## Working style

- One phase per fresh session. Keep context clean.
- Commit after every working phase, but never push. I push to the remote myself. Push only if I explicitly tell you to in that message.
- Commit messages: short, plain, human. One line saying what changed, with a why only when it isn't obvious. No AI boilerplate, no emoji, no padded "feat(scope): comprehensive refactor of ...". Write them the way a developer in a hurry would.
- Code comments: sparing, only where the code isn't self-explanatory. Explain why, not what. Do not narrate obvious lines.
- Always run the tests yourself and show me the output. Never say "this should work."
