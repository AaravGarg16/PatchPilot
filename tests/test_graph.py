import pytest

from agent.graph import build_graph
from agent.schemas import FileEdit, Patch, Plan
from agent.state import PatchPilotState
from tests.fakes import FIXED_PAGINATION, FIXTURE_REPO, HALF_FIXED_PAGINATION, FakeLLM

PLAN = Plan(
    problem="paginate is off by one page and total_pages floors instead of rounding up.",
    relevant_files=["shop/pagination.py", "tests/test_pagination.py", "shop/made_up.py"],
    implementation_steps=["Start pages at (page - 1) * per_page", "Round total_pages up"],
    tests_needed=["tests/test_pagination.py"],
    risks=["Callers relying on 0-indexed pages"],
)


def patch(content: str, path: str = "shop/pagination.py") -> Patch:
    return Patch(summary="Fix pagination", edits=[FileEdit(path=path, content=content)])


def run(llm, tmp_path, **state):
    graph = build_graph(llm, workspace_root=str(tmp_path))
    return graph.invoke(PatchPilotState(repo_id=FIXTURE_REPO, issue="Pagination skips the first page", **state))


def test_fixes_bug_after_one_retry(tmp_path):
    llm = FakeLLM({Plan: [PLAN], Patch: [patch(HALF_FIXED_PAGINATION), patch(FIXED_PAGINATION)]})

    result = run(llm, tmp_path)

    assert result["success"] is True
    assert result["attempt_number"] == 2
    assert result["test_results"].passed
    assert "+    return math.ceil(total_items / per_page)" in result["diff"]
    assert result["modified_files"] == ["shop/pagination.py"]

    # The retry must see why the first attempt failed.
    first, second = llm.prompts_for(Patch)
    assert "Previous attempt failed" not in first
    assert "test_total_pages_rounds_up" in second


def test_planner_sees_paths_not_file_contents(tmp_path):
    llm = FakeLLM({Plan: [PLAN], Patch: [patch(FIXED_PAGINATION)]})

    result = run(llm, tmp_path)

    (planner_prompt,) = llm.prompts_for(Plan)
    assert "shop/pagination.py" in planner_prompt
    assert "def paginate" not in planner_prompt
    # Invented paths from the planner are dropped.
    assert result["plan"].relevant_files == ["shop/pagination.py", "tests/test_pagination.py"]
    (executor_prompt,) = llm.prompts_for(Patch)
    assert "def paginate" in executor_prompt
    assert "format_price" not in executor_prompt


def test_stops_after_max_attempts(tmp_path):
    llm = FakeLLM({Plan: [PLAN], Patch: [patch(HALF_FIXED_PAGINATION)]})

    result = run(llm, tmp_path, max_attempts=2)

    assert result["success"] is False
    assert result["attempt_number"] == 2
    assert len(llm.prompts_for(Patch)) == 2


def test_edits_to_existing_tests_are_rejected(tmp_path):
    cheat = "def test_nothing():\n    assert True\n"
    llm = FakeLLM({Plan: [PLAN], Patch: [patch(cheat, path="tests/test_pagination.py")]})

    result = run(llm, tmp_path, max_attempts=1)

    assert result["success"] is False
    assert result["rejected_edits"] == ["tests/test_pagination.py: existing test files cannot be modified"]
    assert result["diff"] == ""


def test_planner_with_no_real_files_fails_loudly(tmp_path):
    bad_plan = PLAN.model_copy(update={"relevant_files": ["nope.py"]})
    llm = FakeLLM({Plan: [bad_plan], Patch: [patch(FIXED_PAGINATION)]})

    with pytest.raises(ValueError, match="no files that exist"):
        run(llm, tmp_path)
