import logging
import time
import uuid

from pydantic import BaseModel

from agent import target
from agent.errors import describe_error, is_fatal
from agent.graph import _describe, build_graph
from agent.state import AgentConfig, PatchPilotState
from observability.langfuse import trace_run
from observability.usage import UsageTracker
from retrieval.retriever import Retriever
from sandbox.runner import SandboxConfig, destroy_run

logger = logging.getLogger("patchpilot.run")


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
    # Of input_tokens: served from the prompt cache (billed at 0.1x) and written to it (1.25x).
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    # Context utilization: the size of each prompt sent, in input tokens.
    max_prompt_tokens: int
    mean_prompt_tokens: float
    cost_usd: float | None
    latency_seconds: float
    trace_id: str | None
    # Set when the run crashed and raise_errors was False; the metrics then cover what ran before the crash.
    error: str | None = None
    # Recoveries from malformed model output (retries, fallback plan or verdict); empty for a clean run.
    degradations: list[str] = []
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
    agent_config: AgentConfig | None = None,
    on_update=None,
    raise_errors: bool = True,
    **state_fields,
) -> RunReport:
    """Run the agent once, traced end to end, and always clean up the run's container and index.

    With raise_errors=False a crash is recorded on the report instead of raised, so evals keep the cost of failures.
    """
    run_id = uuid.uuid4().hex[:12]
    usage = UsageTracker()
    started = time.monotonic()
    agent_config = agent_config or AgentConfig()
    initial = PatchPilotState(run_id=run_id, repo_id=repo_id, issue=issue, agent_config=agent_config, **state_fields)
    final: dict = {}
    error = None

    with trace_run(run_id, input={"repo": repo_id, "issue": issue}, metadata={"model": model_name}) as trace:
        graph = build_graph(llm, retriever, sandbox_config, workspace_root, agent_config)
        config = {"callbacks": [usage, *trace.callbacks], "run_name": "patchpilot", "metadata": {"run_id": run_id}}
        try:
            async for update in graph.astream(initial, config=config, stream_mode="updates"):
                for node, values in update.items():
                    final.update(values or {})
                    if on_update:
                        on_update(node, final)
        except (Exception, BaseExceptionGroup) as e:
            # Exception groups too: a failure inside the executor's tool session arrives wrapped in one.
            if raise_errors or is_fatal(e):
                raise
            error = describe_error(e)
        finally:
            # finalize does this on the happy path; this covers exceptions and interrupted runs. Best effort: a
            # cleanup problem must not replace the run's result or its real error.
            for cleanup in (lambda: destroy_run(run_id), lambda: retriever.drop(run_id) if agent_config.rag else None):
                try:
                    cleanup()
                except Exception as e:  # noqa: BLE001
                    logger.warning("cleanup after run %s failed: %s", run_id, describe_error(e))

        state = initial.model_copy(update=final)
        report = RunReport(
            run_id=run_id,
            model=model_name,
            success=bool(state.success),
            attempts=state.attempt_number,
            retries=max(state.attempt_number - 1, 0),
            tests_before_passed=state.tests_before.passed if state.tests_before and state.tests_before.ran else None,
            tests_after_passed=state.test_results.passed if state.test_results and state.test_results.ran else None,
            tool_calls=state.tool_calls,
            tool_errors=state.tool_errors,
            llm_calls=usage.calls,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            max_prompt_tokens=max(usage.prompt_tokens, default=0),
            mean_prompt_tokens=round(sum(usage.prompt_tokens) / len(usage.prompt_tokens), 1) if usage.prompt_tokens else 0.0,
            cost_usd=usage.cost_usd(),
            latency_seconds=round(time.monotonic() - started, 2),
            trace_id=trace.trace_id,
            error=error,
            degradations=state.degradations,
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
    print(f"Model: {settings.provider}/{settings.model}; embeddings: {embedder.describe()}")
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
    def status(passed):
        return "not run" if passed is None else ("pass" if passed else "fail")

    print(f"Tests: {status(report.tests_before_passed)} before, {status(report.tests_after_passed)} after")
    print(f"LLM calls: {report.llm_calls}; tokens: {report.input_tokens} in / {report.output_tokens} out; cost: {cost}")
    print(f"Tool calls: {report.tool_calls} ({report.tool_errors} errors); latency: {report.latency_seconds}s")
    if report.trace_id:
        print(f"Langfuse trace: {report.trace_id}")
    print()
    print(report.state.diff or "(no changes)")
