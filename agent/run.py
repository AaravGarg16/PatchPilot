import time
import uuid

from pydantic import BaseModel

from agent import target
from agent.graph import _describe, build_graph
from agent.state import PatchPilotState
from observability.langfuse import trace_run
from observability.usage import UsageTracker
from retrieval.retriever import Retriever
from sandbox.runner import SandboxConfig, destroy_run


class RunReport(BaseModel):
    run_id: str
    model: str
    success: bool
    attempts: int
    retries: int
    tests_before_passed: bool | None
    tests_after_passed: bool | None
    tool_calls: int
    tool_errors: int
    llm_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_seconds: float
    trace_id: str | None
    state: PatchPilotState


async def run_issue(
    repo_id: str,
    issue: str,
    *,
    llm,
    model_name: str,
    retriever: Retriever,
    sandbox_config: SandboxConfig | None = None,
    workspace_root: str | None = None,
    on_update=None,
    **state_fields,
) -> RunReport:
    """Run the agent once, traced end to end, and always clean up the run's container and index."""
    run_id = uuid.uuid4().hex[:12]
    usage = UsageTracker()
    started = time.monotonic()
    initial = PatchPilotState(run_id=run_id, repo_id=repo_id, issue=issue, **state_fields)
    final: dict = {}

    with trace_run(run_id, input={"repo": repo_id, "issue": issue}, metadata={"model": model_name}) as trace:
        graph = build_graph(llm, retriever, sandbox_config, workspace_root)
        config = {"callbacks": [usage, *trace.callbacks], "run_name": "patchpilot", "metadata": {"run_id": run_id}}
        try:
            async for update in graph.astream(initial, config=config, stream_mode="updates"):
                for node, values in update.items():
                    final.update(values or {})
                    if on_update:
                        on_update(node, final)
        finally:
            # finalize does this on the happy path; this covers exceptions and interrupted runs.
            destroy_run(run_id)
            retriever.drop(run_id)

        state = initial.model_copy(update=final)
        report = RunReport(
            run_id=run_id,
            model=model_name,
            success=bool(state.success),
            attempts=state.attempt_number,
            retries=max(state.attempt_number - 1, 0),
            tests_before_passed=state.tests_before.passed if state.tests_before else None,
            tests_after_passed=state.test_results.passed if state.test_results else None,
            tool_calls=state.tool_calls,
            tool_errors=state.tool_errors,
            llm_calls=usage.calls,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=usage.cost_usd(),
            latency_seconds=round(time.monotonic() - started, 2),
            trace_id=trace.trace_id,
            state=state,
        )
        trace.finish(
            output={"success": report.success, "diff": state.diff},
            scores={"success": float(report.success), "attempts": report.attempts, "tool_errors": report.tool_errors},
        )
    return report


async def run_target():
    from agent.llm import get_chat_model, load_llm_settings
    from observability.langfuse import langfuse_enabled
    from retrieval.embeddings import load_embedder
    from retrieval.store import ChunkStore, database_url

    settings = load_llm_settings()
    embedder = load_embedder()
    print(f"Model: {settings.provider}/{settings.model}; embeddings: {embedder.model_name} (local)")
    print(f"Langfuse tracing: {'on' if langfuse_enabled() else 'off (set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)'}")

    def show(node: str, final: dict):
        line = _describe(node, final)
        if line:
            print(f"[{node}] {line}")

    report = await run_issue(
        target.REPO,
        target.ISSUE,
        llm=get_chat_model(settings),
        model_name=settings.model,
        retriever=Retriever(ChunkStore(database_url(), embedder.dim), embedder),
        on_update=show,
    )

    cost = f"${report.cost_usd:.4f}" if report.cost_usd is not None else "unknown (model not in price table)"
    print(f"\nResult: {'SUCCESS' if report.success else 'FAILED'} after {report.attempts} attempt(s)")
    print(f"Tests: {'pass' if report.tests_before_passed else 'fail'} before, {'pass' if report.tests_after_passed else 'fail'} after")
    print(f"LLM calls: {report.llm_calls}; tokens: {report.input_tokens} in / {report.output_tokens} out; cost: {cost}")
    print(f"Tool calls: {report.tool_calls} ({report.tool_errors} errors); latency: {report.latency_seconds}s")
    if report.trace_id:
        print(f"Langfuse trace: {report.trace_id}")
    print()
    print(report.state.diff or "(no changes)")
