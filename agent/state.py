from typing import Optional

from pydantic import BaseModel, Field

from agent.schemas import Plan
from tools.testing import TestResult


class RepoMetadata(BaseModel):
    path: str
    files: list[str]


class PatchPilotState(BaseModel):
    """Shared LangGraph state. Later phases add retrieval, lint, evaluation and cost fields."""

    repo_id: Optional[str] = None
    issue: Optional[str] = None
    repo_metadata: Optional[RepoMetadata] = None

    plan: Optional[Plan] = None

    modified_files: list[str] = Field(default_factory=list)
    rejected_edits: list[str] = Field(default_factory=list)
    patch_summary: Optional[str] = None
    diff: Optional[str] = None
    test_results: Optional[TestResult] = None
    success: Optional[bool] = None

    attempt_number: int = 0
    max_attempts: int = 3
