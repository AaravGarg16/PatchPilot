"""Langfuse tracing for a whole run.

A no-op unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set. LANGFUSE_TRACING_ENABLED=false (Langfuse's own
switch) turns it off even when keys are present; the test suite uses that so it never traces to a real project.
"""
import os
from collections.abc import Iterator
from contextlib import contextmanager

from dotenv import load_dotenv


def langfuse_enabled() -> bool:
    load_dotenv()
    if os.getenv("LANGFUSE_TRACING_ENABLED", "true").strip().lower() in ("false", "0", "no"):
        return False
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


class RunTrace:
    """Handle for one run's trace: LangChain callbacks to attach, and a place to record the outcome."""

    def __init__(self, callbacks: list | None = None, span=None, trace_id: str | None = None):
        self.callbacks = callbacks or []
        self._span = span
        self.trace_id = trace_id

    def finish(self, output: dict, scores: dict[str, float]) -> None:
        if self._span is None:
            return
        self._span.update(output=output)
        for name, value in scores.items():
            self._span.score_trace(name=name, value=value)


@contextmanager
def trace_run(run_id: str, input: dict, metadata: dict[str, str]) -> Iterator[RunTrace]:
    """Open a root span for the run. Every LangGraph node, LLM call and MCP tool call nests under it via the callback."""
    if not langfuse_enabled():
        yield RunTrace()
        return

    from langfuse import Langfuse, propagate_attributes
    from langfuse.langchain import CallbackHandler

    client = Langfuse()
    try:
        with (
            client.start_as_current_observation(name="patchpilot-run", as_type="span", input=input) as span,
            # session_id groups a run's trace; tags and metadata make runs filterable in the Langfuse UI.
            propagate_attributes(session_id=run_id, trace_name="patchpilot-run", tags=["patchpilot"], metadata=metadata),
        ):
            yield RunTrace([CallbackHandler()], span, client.get_current_trace_id())
    finally:
        client.flush()  # spans export in the background; make sure this run's are sent before returning
