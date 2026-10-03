
from pydantic import BaseModel, Field

from agent.schemas import Evaluation, Plan
from tools.linting import LintResult
from tools.testing import TestResult


class RepoMetadata(BaseModel):
    path: str
    files: list[str]


class PatchPilotState(BaseModel):
    """Shared LangGraph state. Later phases add retrieval and cost fields."""

    repo_id: str | None = None
    issue: str | None = None
    repo_metadata: RepoMetadata | None = None

    plan: Plan | None = None

    modified_files: list[str] = Field(default_factory=list)
    patch_summary: str | None = None
    diff: str | None = None
    test_results: TestResult | None = None
    lint_results: LintResult | None = None

    evaluation: Evaluation | None = None
    failure_history: list[str] = Field(default_factory=list)
    success: bool | None = None

    attempt_number: int = 0
    max_attempts: int = 3

    tool_calls: int = 0
    tool_errors: int = 0
