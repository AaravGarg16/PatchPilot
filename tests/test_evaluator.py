"""Evaluator calibration: when passing tests count, and when the evaluator can still reject."""
import pytest

from agent.evaluator import evaluate, is_accepted
from agent.schemas import Evaluation, Plan
from agent.state import PatchPilotState, RepoMetadata
from tests.fakes import FIXED_PAGINATION, FIXTURE_REPO, FakeLLM
from tools import filesystem, git
from tools.linting import LintResult
from tools.testing import TestResult

PLAN = Plan(problem="Pagination is off by one.", relevant_files=["shop/pagination.py"],
            implementation_steps=["Fix the offset"], tests_needed=["tests/test_pagination.py"], risks=[])
PASSED = TestResult(passed=True, exit_code=0, output="12 passed", selected=["tests/test_pagination.py"])
FAILED = TestResult(passed=False, exit_code=1, output="1 failed, 11 passed", selected=["tests/test_pagination.py"])
NOT_RUN = TestResult(passed=False, exit_code=None, output="no related test files", ran=False)


def verdict(correct=True, requirements_met=True, action="finish", problems=()) -> Evaluation:
    return Evaluation(correct=correct, requirements_met=requirements_met, tests_pass=True,
                      problems=list(problems), recommended_action=action)


@pytest.fixture
def state(tmp_path):
    made = iter(range(100))

    def make(tests: TestResult, patched: bool = True) -> PatchPilotState:
        repo = git.checkout_repo(FIXTURE_REPO, tmp_path / f"repo{next(made)}")
        if patched:
            filesystem.write_file(repo, "shop/pagination.py", FIXED_PAGINATION)
        return PatchPilotState(
            issue="Pagination skips the first page", plan=PLAN, attempt_number=1, patch_summary="Fixed the offset.",
            repo_metadata=RepoMetadata(path=str(repo), files=git.list_files(repo)),
            test_results=tests, lint_results=LintResult(passed=True, output=""),
        )
    return make


def run(state: PatchPilotState, evaluation: Evaluation) -> tuple[dict, FakeLLM]:
    llm = FakeLLM({Evaluation: [evaluation]})
    return evaluate(state, llm), llm


def test_passing_tests_are_not_overruled_by_a_retry_without_a_real_gap(state):
    """The django-11179 case: target tests pass, the evaluator calls the patch correct but still says retry."""
    update, _ = run(state(PASSED), verdict(action="retry", problems=["Add a regression test for the fix."]))
    assert update["success"] is True
    assert "failure_history" not in update  # no wasted attempt


def test_passing_tests_are_not_overruled_by_a_rejection_that_names_nothing(state):
    update, _ = run(state(PASSED), verdict(correct=False, requirements_met=False, action="retry"))
    assert update["success"] is True


def test_a_concrete_gap_still_rejects_a_patch_whose_tests_pass(state):
    gap = "The issue also asks for total_pages to round up; the diff leaves it flooring."
    update, _ = run(state(PASSED), verdict(requirements_met=False, action="retry", problems=[gap]))
    assert update["success"] is False
    (record,) = update["failure_history"]
    assert record.problems == [gap] and record.tests_ran and record.tests_passed
    assert record.tests_selected == ["tests/test_pagination.py"]


def test_failing_tests_are_never_accepted(state):
    update, _ = run(state(FAILED), verdict())
    assert update["success"] is False


def test_an_empty_diff_is_never_accepted(state):
    update, _ = run(state(PASSED, patched=False), verdict())
    assert update["success"] is False


def test_without_tests_the_evaluator_decides(state):
    assert run(state(NOT_RUN), verdict())[0]["success"] is True
    assert run(state(NOT_RUN), verdict(action="retry", problems=["x"]))[0]["success"] is False
    # With no tests, "correct" alone is not enough; it has to recommend finish.
    assert run(state(NOT_RUN), verdict(action="retry"))[0]["success"] is False


def test_the_evaluator_is_told_what_does_not_count_as_a_gap(state):
    _, llm = run(state(PASSED), verdict())
    (messages,) = [m for s, m in llm.structured_calls if s is Evaluation]
    system, human = messages[0][1], messages[-1][1]
    assert "the executor cannot edit existing test files" in system
    assert "Tests: PASSED (tests/test_pagination.py)" in human


@pytest.mark.parametrize(
    ("tests", "evaluation", "accepted"),
    [
        (PASSED, verdict(), True),
        (PASSED, verdict(action="retry", problems=["style nit"]), True),
        (PASSED, verdict(correct=False, action="retry", problems=["the diff breaks page 0 handling"]), False),
        (FAILED, verdict(), False),
        (NOT_RUN, verdict(correct=False, action="finish"), False),
    ],
)
def test_acceptance_rule(state, tests, evaluation, accepted):
    s = state(tests)
    assert is_accepted(s, evaluation, git.diff(git.Path(s.repo_metadata.path))) is accepted
