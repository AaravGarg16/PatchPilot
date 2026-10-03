from pydantic import BaseModel, Field

from agent.context import DEFAULT_BUDGET_TOKENS
from agent.schemas import Evaluation, Plan
from retrieval.retriever import ScoredChunk
from tools.linting import LintResult
from tools.testing import TestResult


class RepoMetadata(BaseModel):
    path: str
    files: list[str]


class FailureRecord(BaseModel):
    attempt: int
    tests_passed: bool
    test_output: str
    lint_passed: bool
    lint_output: str
    problems: list[str]


class PatchPilotState(BaseModel):
    """Shared LangGraph state. Run-level metrics (tokens, cost, latency) live on RunReport in agent/run.py."""

    run_id: str | None = None
    repo_id: str | None = None
    issue: str | None = None
    repo_metadata: RepoMetadata | None = None

    sandbox_id: str | None = None
    tests_before: TestResult | None = None

    index_id: str | None = None
    indexed_chunks: int = 0
    retrieved_chunks: list[ScoredChunk] = Field(default_factory=list)
    retrieved_files: list[str] = Field(default_factory=list)
    context_budget: int = DEFAULT_BUDGET_TOKENS

    plan: Plan | None = None

    modified_files: list[str] = Field(default_factory=list)
    patch_summary: str | None = None
    diff: str | None = None
    test_results: TestResult | None = None
    lint_results: LintResult | None = None

    evaluation: Evaluation | None = None
    failure_history: list[FailureRecord] = Field(default_factory=list)
    success: bool | None = None

    attempt_number: int = 0
    max_attempts: int = 3

    tool_calls: int = 0
    tool_errors: int = 0
    elided_tool_outputs: int = 0
