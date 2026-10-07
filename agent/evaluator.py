from pathlib import Path

from agent.context import ContextBudget, fit, render_failures
from agent.prompts import evaluator as prompts
from agent.schemas import Evaluation
from agent.state import FailureRecord, PatchPilotState
from tools import git


def _test_status(tests) -> str:
    if not tests.ran:
        return "NOT RUN (no related test files found; judge the diff on its own)"
    return f"{'PASSED' if tests.passed else 'FAILED'} ({', '.join(tests.selected) or 'full suite'})"


def tests_ok(state: PatchPilotState) -> bool:
    """Real test results gate acceptance. When nothing could be run there is no signal either way."""
    return state.test_results.passed if state.test_results.ran else True


def is_accepted(state: PatchPilotState, evaluation: Evaluation, diff: str) -> bool:
    """Combine real test results with the evaluator's judgment.

    Passing selected tests are evidence the model alone cannot overrule: a rejection then needs a concrete problem
    and a verdict that the patch is wrong or incomplete. A vague or inconsistent rejection ("correct, but retry",
    or "incorrect" with nothing named) does not cost another attempt. Failing tests and empty diffs are never accepted.
    """
    if not diff.strip():
        return False
    tests = state.test_results
    if tests.ran and not tests.passed:
        return False
    if tests.ran:
        names_a_gap = bool(evaluation.problems) and not (evaluation.correct and evaluation.requirements_met)
        return not names_a_gap
    # Nothing ran, so the evaluator's reading of the diff is the only signal.
    return evaluation.correct and evaluation.recommended_action == "finish"


def _failure_record(state: PatchPilotState, problems: list[str]) -> FailureRecord:
    return FailureRecord(
        attempt=state.attempt_number,
        tests_passed=state.test_results.passed,
        tests_ran=state.test_results.ran,
        tests_selected=state.test_results.selected,
        test_output=state.test_results.output,
        lint_passed=state.lint_results.passed,
        lint_output=state.lint_results.output,
        problems=problems,
    )


def build_evaluator_message(state: PatchPilotState, diff: str) -> str:
    budget = ContextBudget(total_tokens=state.context_budget)
    tests, lint = state.test_results, state.lint_results
    output_tokens = budget.tokens("tool_outputs")
    parts = [
        f"## Issue\n{fit(state.issue, budget.tokens('task'))}",
        f"## Plan\n{fit(state.plan.model_dump_json(indent=2), budget.tokens('plan')) if state.plan else '(no plan)'}",
        f"## Executor summary\n{fit(state.patch_summary or '', 500)}",
        f"## Diff\n```diff\n{fit(diff, budget.tokens('code')) or '(no changes)'}\n```",
        f"## Tests: {_test_status(tests)}\n```\n{fit(tests.output, output_tokens * 3 // 4, keep='tail')}\n```",
        (
            f"## Lint (whole changed files, so some problems may predate the patch): {'CLEAN' if lint.passed else 'PROBLEMS'}\n"
            f"```\n{fit(lint.output, output_tokens // 4, keep='tail')}\n```"
        ),
    ]
    if state.failure_history:
        parts.append("## Earlier failed attempts\n" + render_failures(state.failure_history, budget.tokens("failures")))
    return "\n\n".join(parts)


def evaluate(state: PatchPilotState, llm) -> dict:
    diff = git.diff(Path(state.repo_metadata.path))
    messages = [("system", prompts.SYSTEM), ("human", build_evaluator_message(state, diff))]
    run_config = {"run_name": f"evaluator attempt {state.attempt_number}", "metadata": {"attempt": state.attempt_number}}
    evaluation: Evaluation = llm.with_structured_output(Evaluation, method="json_schema").invoke(messages, config=run_config)

    accepted = is_accepted(state, evaluation, diff)
    update = {"evaluation": evaluation, "diff": diff, "success": accepted}
    if not accepted:
        update["failure_history"] = [*state.failure_history, _failure_record(state, evaluation.problems)]
    return update


def judge_by_tests(state: PatchPilotState) -> dict:
    """The no-evaluator ablation: accept on test results alone."""
    diff = git.diff(Path(state.repo_metadata.path))
    accepted = tests_ok(state) and bool(diff)
    update = {"diff": diff, "success": accepted}
    if not accepted:
        problems = [] if diff else ["No changes were made."]
        update["failure_history"] = [*state.failure_history, _failure_record(state, problems)]
    return update
