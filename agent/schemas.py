from typing import Literal

from pydantic import BaseModel, Field


class Plan(BaseModel):
    problem: str = Field(description="What is wrong, in one or two sentences.")
    relevant_files: list[str] = Field(
        description="Repository paths, copied exactly from the file list, that need changes or show expected behavior (include the relevant tests)."
    )
    implementation_steps: list[str]
    tests_needed: list[str] = Field(description="Tests that should pass once the fix is in.")
    risks: list[str]


class Evaluation(BaseModel):
    correct: bool = Field(description="The diff is a correct fix for the issue.")
    requirements_met: bool = Field(description="Every behavior the issue asks for is addressed.")
    tests_pass: bool
    problems: list[str] = Field(description="Concrete, actionable problems for the executor to fix. Empty if none.")
    recommended_action: Literal["finish", "retry"]
