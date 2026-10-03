from pathlib import Path

from agent.feedback import tail
from agent.prompts import evaluator as prompts
from agent.schemas import Evaluation
from agent.state import PatchPilotState
from tools import git


def build_evaluator_message(state: PatchPilotState, diff: str) -> str:
    tests, lint = state.test_results, state.lint_results
    parts = [
        f"## Issue\n{state.issue}",
        f"## Plan\n{state.plan.model_dump_json(indent=2)}",
        f"## Executor summary\n{state.patch_summary}",
        f"## Diff\n```diff\n{tail(diff) or '(no changes)'}\n```",
        f"## Tests: {'PASSED' if tests.passed else 'FAILED'}\n```\n{tail(tests.output)}\n```",
        f"## Lint (whole changed files, so some problems may predate the patch): {'CLEAN' if lint.passed else 'PROBLEMS'}\n```\n{tail(lint.output)}\n```",
    ]
    if state.failure_history:
        parts.append("## Earlier failed attempts\n" + "\n\n".join(state.failure_history))
    return "\n\n".join(parts)


def summarize_failure(state: PatchPilotState, evaluation: Evaluation) -> str:
    tests = state.test_results
    test_line = "passed" if tests.passed else f"failed:\n{tail(tests.output, 2000)}"
    problems = "\n".join(f"- {p}" for p in evaluation.problems) or "- (evaluator listed none)"
    lint = "" if state.lint_results.passed else f"\nLint:\n{tail(state.lint_results.output, 1000)}"
    return f"### Attempt {state.attempt_number}\nTests {test_line}{lint}\nEvaluator problems:\n{problems}"


def evaluate(state: PatchPilotState, llm) -> dict:
    diff = git.diff(Path(state.repo_metadata.path))
    messages = [("system", prompts.SYSTEM), ("human", build_evaluator_message(state, diff))]
    evaluation: Evaluation = llm.with_structured_output(Evaluation, method="json_schema").invoke(messages)

    # Real test results overrule the model: a patch with failing tests is never accepted.
    accepted = state.test_results.passed and evaluation.correct and evaluation.recommended_action == "finish"
    update = {"evaluation": evaluation, "diff": diff, "success": accepted}
    if not accepted:
        update["failure_history"] = [*state.failure_history, summarize_failure(state, evaluation)]
    return update
