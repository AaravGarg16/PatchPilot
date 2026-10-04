import asyncio
import itertools
import tempfile
import uuid
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from agent.evaluator import evaluate, judge_by_tests
from agent.executor import implement
from agent.planner import plan
from agent.state import AgentConfig, IndexStats, PatchPilotState, RepoMetadata
from retrieval.retriever import IndexReport, Retriever, select_files
from sandbox.runner import Sandbox, SandboxConfig
from tools import git, linting, testing
from tools.filesystem import is_test_file

RETRIEVE_TOP_K = 20
MAX_SELECTED_FILES = 5
# Test modules indexed per lookup. Tests are indexed on demand because they are most of a large repo (65% of
# Django's chunks) and only the ones near the code being changed help retrieval.
MAX_INDEXED_TEST_FILES = 8


def analyze_repository(state: PatchPilotState, workspace_root: str | None) -> dict:
    run_id = state.run_id or uuid.uuid4().hex[:12]
    if state.prepared_workspace:
        repo = Path(state.prepared_workspace)
    else:
        workspace = Path(tempfile.mkdtemp(prefix=f"patchpilot-{run_id}-", dir=workspace_root))
        # Cloning is safe on the host: git does not run repository code (hooks are never cloned).
        repo = git.checkout_repo(state.repo_id, workspace / "repo")
    return {
        "run_id": run_id,
        "index_id": run_id,
        "repo_metadata": RepoMetadata(path=str(repo), files=git.list_files(repo)),
    }


def prepare_sandbox(state: PatchPilotState, config: SandboxConfig | None) -> dict:
    """Start the run's container (installing dependencies, then cutting the network) and record the baseline tests."""
    sandbox = Sandbox.start(Path(state.repo_metadata.path), state.run_id, config)
    # Small repos get a full baseline run. Large ones have nothing selected yet, so this records "not run".
    return {"sandbox_id": sandbox.name, "tests_before": testing.run_tests(sandbox, state.checks.test_command)}


def source_files(files: list[str]) -> list[str]:
    """Everything indexed up front: code, docs and config, but no test files."""
    return [f for f in files if not is_test_file(f)]


def index_source(retriever: Retriever, index_id: str, repo: Path, files: list[str]) -> IndexStats:
    report = retriever.index_paths(index_id, repo, source_files(files))
    return IndexStats(source_chunks=report.chunks, skipped_chunks=report.skipped_chunks)


def index_related_tests(
    retriever: Retriever, index_id: str, repo: Path, files: list[str], anchors: list[str], stats: IndexStats
) -> IndexStats:
    """Index the test modules related to the anchor source files that are not indexed yet."""
    related = testing.select_tests(repo, files, anchors, limit=MAX_INDEXED_TEST_FILES)
    new = [t for t in related if t not in stats.test_files]
    if not new:
        return stats
    report = retriever.index_paths(index_id, repo, new)
    merged = IndexReport(skipped_chunks=stats.skipped_chunks).merge(report)
    return stats.model_copy(update={
        "test_chunks": stats.test_chunks + report.chunks,
        "skipped_chunks": merged.skipped_chunks,
        "test_files": [*stats.test_files, *new],
    })


def index_repository(state: PatchPilotState, retriever: Retriever) -> dict:
    stats = index_source(retriever, state.index_id, Path(state.repo_metadata.path), state.repo_metadata.files)
    return {"index_stats": stats, "indexed_chunks": stats.source_chunks}


def _retrieve(retriever: Retriever, state: PatchPilotState, query: str, anchors: list[str]) -> dict:
    """Search, pulling in the tests related to `anchors` plus whatever the first search found, then search again."""
    repo = Path(state.repo_metadata.path)
    found = select_files(retriever.search(state.index_id, query, top_k=RETRIEVE_TOP_K), MAX_SELECTED_FILES)
    sources = [f for f in dict.fromkeys([*anchors, *found]) if not is_test_file(f)]
    stats = index_related_tests(retriever, state.index_id, repo, state.repo_metadata.files, sources, state.index_stats)
    chunks = retriever.search(state.index_id, query, top_k=RETRIEVE_TOP_K)
    return {
        "retrieved_chunks": chunks,
        "retrieved_files": select_files(chunks, MAX_SELECTED_FILES),
        "index_stats": stats,
        "indexed_chunks": stats.source_chunks + stats.test_chunks,
    }


def retrieve_context(state: PatchPilotState, retriever: Retriever) -> dict:
    return _retrieve(retriever, state, state.issue, anchors=[])


def refresh_context(state: PatchPilotState, retriever: Retriever) -> dict:
    """Before a retry: re-index edited files so chunks match the code, index tests related to what changed, and
    search again using the evaluator's problems."""
    repo = Path(state.repo_metadata.path)
    retriever.index_paths(state.index_id, repo, git.changed_files(repo))
    problems = "\n".join(state.failure_history[-1].problems) if state.failure_history else ""
    return _retrieve(retriever, state, f"{state.issue}\n{problems}".strip(), anchors=state.modified_files)


def _test_hints(state: PatchPilotState) -> list[str]:
    planned = [*state.plan.tests_needed, *state.plan.relevant_files] if state.plan else []
    # tests_needed entries can be node ids like tests/test_x.py::test_y
    return [h.split("::")[0] for h in [*planned, *state.modified_files]]


def run_checks(state: PatchPilotState, config: SandboxConfig | None) -> dict:
    # The executor can run these itself through MCP; this run is the authoritative one.
    sandbox = Sandbox.attach(state.sandbox_id, config)
    command = state.checks.test_command
    repo = Path(state.repo_metadata.path)
    paths = [] if command.run_all_when_unselected else testing.select_tests(repo, state.repo_metadata.files, _test_hints(state))
    if state.checks.lint:
        lint = linting.run_linter(sandbox, repo, state.modified_files)
    else:
        lint = linting.LintResult(passed=True, output="linting is disabled for this repository")
    return {"test_results": testing.run_tests(sandbox, command, paths), "lint_results": lint}


def route_after_evaluation(state: PatchPilotState) -> str:
    if state.success or state.attempt_number >= state.max_attempts:
        return "finalize"
    return "retry"


def finalize(state: PatchPilotState, retriever: Retriever) -> dict:
    Sandbox.attach(state.sandbox_id).destroy()
    if state.index_id:
        retriever.drop(state.index_id)
    return {"diff": git.diff(Path(state.repo_metadata.path))}


def build_graph(
    llm,
    retriever: Retriever,
    sandbox_config: SandboxConfig | None = None,
    workspace_root: str | None = None,
    config: AgentConfig | None = None,
):
    """Wire the workflow. Components switched off in `config` are left out of the graph entirely."""
    config = config or AgentConfig()

    async def implement_node(state: PatchPilotState) -> dict:
        return await implement(state, llm)

    graph = StateGraph(PatchPilotState)
    graph.add_node("analyze_repository", lambda s: analyze_repository(s, workspace_root))
    graph.add_node("prepare_sandbox", lambda s: prepare_sandbox(s, sandbox_config))
    graph.add_node("implement", implement_node)
    graph.add_node("run_checks", lambda s: run_checks(s, sandbox_config))
    graph.add_node("finalize", lambda s: finalize(s, retriever))

    # Each step is a node name; the chain is linear apart from the retry loop.
    chain = ["analyze_repository", "prepare_sandbox"]
    if config.rag:
        graph.add_node("index_repository", lambda s: index_repository(s, retriever))
        graph.add_node("retrieve_context", lambda s: retrieve_context(s, retriever))
        graph.add_node("refresh_context", lambda s: refresh_context(s, retriever))
        chain += ["index_repository", "retrieve_context"]
    if config.planner:
        graph.add_node("plan", lambda s: plan(s, llm))
        chain.append("plan")
    judge = "evaluate" if config.evaluator else "judge_by_tests"
    graph.add_node(judge, (lambda s: evaluate(s, llm)) if config.evaluator else judge_by_tests)
    chain += ["implement", "run_checks", judge]

    graph.add_edge(START, chain[0])
    for a, b in itertools.pairwise(chain):
        graph.add_edge(a, b)
    retry_target = "refresh_context" if config.rag else "implement"
    graph.add_conditional_edges(judge, route_after_evaluation, {"retry": retry_target, "finalize": "finalize"})
    if config.rag:
        graph.add_edge("refresh_context", "implement")
    graph.add_edge("finalize", END)
    return graph.compile()


def _describe(node: str, final: dict) -> str | None:
    if node == "analyze_repository":
        return f"run {final['run_id']}, workspace {final['repo_metadata'].path}"
    if node == "prepare_sandbox":
        before = final["tests_before"]
        status = "not run" if not before.ran else ("passing" if before.passed else "failing")
        return f"container {final['sandbox_id']}; tests before the fix: {status}"
    if node == "index_repository":
        stats = final["index_stats"]
        skipped = sum(stats.skipped_chunks.values())
        return f"{stats.source_chunks} source chunks ({skipped} noise chunks skipped)"
    if node in ("retrieve_context", "refresh_context"):
        stats = final["index_stats"]
        return (f"{len(final['retrieved_chunks'])} chunks; top files: {', '.join(final['retrieved_files'])}\n"
                f"        related tests indexed: {', '.join(stats.test_files) or 'none'}")
    if node == "plan":
        return f"{final['plan'].problem}\n        files: {', '.join(final['plan'].relevant_files)}"
    if node == "implement":
        return (
            f"attempt {final['attempt_number']}: {final['patch_summary']}\n"
            f"        tool calls so far: {final['tool_calls']} ({final['tool_errors']} errors, "
            f"{final['elided_tool_outputs']} outputs elided)"
        )
    if node == "run_checks":
        tests, lint = final["test_results"], final["lint_results"]
        last_line = tests.output.strip().splitlines()[-1] if tests.output.strip() else ""
        status = "not run" if not tests.ran else ("passed" if tests.passed else "failed")
        return f"tests {status}: {last_line}; lint {'clean' if lint.passed else 'problems'}"
    if node == "judge_by_tests":
        return "accepted" if final["success"] else "rejected"
    if node == "evaluate":
        ev = final["evaluation"]
        lines = [f"{ev.recommended_action} (correct={ev.correct}, requirements_met={ev.requirements_met})"]
        return "\n".join([*lines, *(f"        - {p}" for p in ev.problems)])
    return None


def main():
    from agent.run import run_target

    asyncio.run(run_target())


if __name__ == "__main__":
    main()
