from pydantic import BaseModel, Field


class Plan(BaseModel):
    problem: str = Field(description="What is wrong, in one or two sentences.")
    relevant_files: list[str] = Field(
        description="Repository paths, copied exactly from the file list, that need changes or show expected behavior (include the relevant tests)."
    )
    implementation_steps: list[str]
    tests_needed: list[str] = Field(description="Tests that should pass once the fix is in.")
    risks: list[str]


class FileEdit(BaseModel):
    path: str = Field(description="Repository-relative path of the file to write.")
    content: str = Field(description="The complete new content of the file, not a fragment.")


class Patch(BaseModel):
    summary: str = Field(description="One or two sentences on what the edits change and why.")
    edits: list[FileEdit]
