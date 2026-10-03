from pathlib import Path, PurePosixPath

from agent.prompts import executor as prompts
from agent.schemas import Patch
from agent.state import PatchPilotState
from tools import filesystem

# Keeps prompts bounded when a test run is noisy; failures are summarized at the end of pytest output.
MAX_TEST_OUTPUT_CHARS = 6000


def is_test_file(rel_path: str) -> bool:
    p = PurePosixPath(rel_path)
    return (
        "tests" in p.parts
        or "test" in p.parts
        or p.name.startswith("test_")
        or p.name.endswith("_test.py")
        or p.name == "conftest.py"
    )


def build_executor_message(state: PatchPilotState, files: dict[str, str]) -> str:
    parts = [f"## Issue\n{state.issue}", f"## Plan\n{state.plan.model_dump_json(indent=2)}"]
    for path, content in files.items():
        parts.append(f"## File: {path}\n```\n{content}\n```")

    if state.test_results and not state.test_results.passed:
        output = state.test_results.output
        if len(output) > MAX_TEST_OUTPUT_CHARS:
            output = "[earlier output truncated]\n" + output[-MAX_TEST_OUTPUT_CHARS:]
        parts.append(
            f"## Previous attempt failed (attempt {state.attempt_number} of {state.max_attempts})\n"
            f"Test output:\n```\n{output}\n```"
        )
    if state.rejected_edits:
        parts.append("## Edits rejected last attempt\n" + "\n".join(f"- {r}" for r in state.rejected_edits))
    return "\n\n".join(parts)


def apply_patch(repo: Path, patch: Patch) -> tuple[list[str], list[str]]:
    """Write the edits. Returns (modified paths, rejection reasons)."""
    root = repo.resolve()
    modified, rejected = [], []
    for edit in patch.edits:
        try:
            target = filesystem.resolve_in_repo(repo, edit.path)
        except ValueError as e:
            rejected.append(str(e))
            continue
        rel = target.relative_to(root).as_posix()
        # Rewriting an existing test is the easiest way to "pass", so it is never allowed.
        if target.exists() and is_test_file(rel):
            rejected.append(f"{rel}: existing test files cannot be modified")
            continue
        filesystem.write_file(repo, rel, edit.content)
        modified.append(rel)
    return modified, rejected


def implement(state: PatchPilotState, llm) -> dict:
    repo = Path(state.repo_metadata.path)
    files = {
        path: filesystem.read_file(repo, path)
        for path in state.plan.relevant_files
        if filesystem.resolve_in_repo(repo, path).is_file()
    }
    messages = [("system", prompts.SYSTEM), ("human", build_executor_message(state, files))]
    patch: Patch = llm.with_structured_output(Patch, method="json_schema").invoke(messages)

    modified, rejected = apply_patch(repo, patch)
    return {
        "attempt_number": state.attempt_number + 1,
        "modified_files": sorted(set(state.modified_files) | set(modified)),
        "rejected_edits": rejected,
        "patch_summary": patch.summary,
    }
