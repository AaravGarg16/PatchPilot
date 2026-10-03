from typing import Optional

from pydantic import BaseModel


class PatchPilotState(BaseModel):
    """Shared LangGraph state. Only the fields the Phase 2 core loop needs; later phases extend it."""

    repo_id: Optional[str] = None
    issue: Optional[str] = None
    repo_metadata: Optional[dict] = None

    # Becomes a typed Plan schema once the planner exists in Phase 2.
    plan: Optional[dict] = None

    modified_files: Optional[list[str]] = None
    diff: Optional[str] = None
    test_results: Optional[dict] = None

    attempt_number: Optional[int] = None
    max_attempts: Optional[int] = None
