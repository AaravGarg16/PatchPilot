from typing import Literal

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field

from agent.context import DEFAULT_BUDGET_TOKENS
from agent.schemas import Evaluation, Plan
from retrieval.retriever import ScoredChunk
from tools.linting import LintResult
from tools.testing import TestCommand, TestResult


class AgentConfig(BaseModel):
    """Which components run. The eval ablation turns these off one at a time; everything is on by default."""

    rag: bool = True
    planner: bool = True
    evaluator: bool = True
    # "full_history" is the context-engineering baseline: no budget, no elision, every attempt appends to one transcript.
    context_mode: Literal["dynamic", "full_history"] = "dynamic"


class CheckSettings(BaseModel):
    """How this repo is tested and linted. Defaults suit small pytest repos; SWE-bench tasks override them."""

    test_command: TestCommand = TestCommand()
    lint: bool = True


class IndexStats(BaseModel):
    """What the run's index holds: source up front, test modules only when related to the code in play."""

    source_chunks: int = 0
    test_chunks: int = 0
    # Chunks that noise files would have added, by reason (too_large, long_lines, generated, data).
    skipped_chunks: dict[str, int] = {}
    test_files: list[str] = []


class RepoMetadata(BaseModel):
    path: str
    files: list[str]


class FailureRecord(BaseModel):
    attempt: int
    tests_passed: bool
    tests_ran: bool = True
    tests_selected: list[str] = []
    test_output: str
    lint_passed: bool
    lint_output: str
    problems: list[str]


class PatchPilotState(BaseModel):
    """Shared LangGraph state. Run-level metrics (tokens, cost, latency) live on RunReport in agent/run.py."""

    run_id: str | None = None
    repo_id: str | None = None
    issue: str | None = None
    # A workspace someone else already set up (SWE-bench copies one out of the task image). Otherwise repo_id is cloned.
    prepared_workspace: str | None = None
    repo_metadata: RepoMetadata | None = None

    agent_config: AgentConfig = AgentConfig()
    checks: CheckSettings = CheckSettings()

    sandbox_id: str | None = None
    tests_before: TestResult | None = None

    index_id: str | None = None
    indexed_chunks: int = 0
    index_stats: IndexStats = IndexStats()
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

    # Recoveries from malformed model output, each prefixed "retry:" (a second ask worked) or "fallback:" (a step's
    # result was replaced).
    degradations: list[str] = Field(default_factory=list)

    tool_calls: int = 0
    tool_errors: int = 0
    elided_tool_outputs: int = 0
    # Only kept in full_history mode, where each attempt continues the previous transcript.
    executor_history: list[AnyMessage] = Field(default_factory=list)
