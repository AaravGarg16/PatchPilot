"""Export the featured runs as static JSON for the replay UI. Reads only local results and the cached dataset.

    python -m evals.export_replays      # writes apps/web/src/data/replays.json

Everything in the output comes from a recorded run or the SWE-bench dataset. Result files hold outcomes rather than
steps, so what a run did not record (the plan's text, individual tool calls, test output) is left out, not invented.
"""
import json

from evals.results import TaskResult
from evals.tasks import EVALS_DIR, PROJECT_ROOT, SWEBENCH_LITE

OUTPUT = PROJECT_ROOT / "apps" / "web" / "src" / "data" / "replays.json"
MAX_ATTEMPTS = 3  # the limit the benchmark ran with

CONFIGS = {
    "planner": {
        "label": "Retrieval + planner",
        "components": ["Retrieval", "Planner", "Executor", "Sandboxed tests"],
        "judge": "tests",
        "note": "This run used the pipeline without the LLM evaluator: a patch is accepted when the related tests pass.",
    },
    "full": {
        "label": "Full pipeline",
        "components": ["Retrieval", "Planner", "Executor", "Sandboxed tests", "Evaluator"],
        "judge": "evaluator",
        "note": "The full pipeline: after each attempt an LLM evaluator reviews the diff and the test results.",
    },
}

# (results folder, config, instance id, tag, takeaway). The takeaway states only what the run's data shows.
# The first entry is the replay the page opens on.
FEATURED = [
    ("bench", "full", "django__django-11049", "Retry loop",
     ("Three attempts. Each time the related tests failed on an existing test that still asserted the old message, "
     "and the evaluator sent written feedback back to the executor. The agent is not allowed to edit existing tests, "
     "so it hit its attempt limit without accepting its own patch. The official grader, which runs the updated "
     "tests, marked the patch resolved.")),
    ("bench", "planner", "django__django-13401", "Single attempt",
     ("One attempt, six tool calls. The planner narrowed five retrieved files to the one that mattered and its tests; "
     "the executor rewrote __eq__, __lt__ and __hash__ on Field. The official grader's hidden test passed, with all "
     "32 existing tests still passing.")),
    ("bench", "planner", "sympy__sympy-18835", "Different codebase",
     ("A different repository and a behavior change rather than a one-line fix: uniq() now raises RuntimeError when "
     "the list it iterates is modified, as the issue asked. One attempt, 21 tool calls, and the official grader's "
     "hidden test passed alongside 41 existing ones.")),
]


def _attempts(result: TaskResult, judge: str) -> list[dict]:
    attempts = [
        {
            "number": r["attempt"],
            "tests": {"ran": r["tests_ran"], "passed": r["tests_passed"], "selected": r["tests_selected"]},
            "accepted": False,
            "judge": judge,
            "problems": r["problems"],
        }
        for r in result.rejections
    ]
    if result.self_accepted:
        problems = (result.final_evaluation or {}).get("problems", [])
        attempts.append({
            "number": result.attempts,
            # Which test files ran is only recorded for rejected attempts.
            "tests": {"ran": result.tests_after_passed is not None, "passed": bool(result.tests_after_passed), "selected": None},
            "accepted": True,
            "judge": judge,
            "problems": problems,
        })
    return attempts


def build_replay(folder: str, config: str, instance_id: str, tag: str, takeaway: str, issues: dict[str, dict]) -> dict:
    path = EVALS_DIR / "results" / folder / config / "tasks" / f"{instance_id}.json"
    result = TaskResult.model_validate_json(path.read_text())
    if not (result.grade and result.grade.resolved):
        raise ValueError(f"{instance_id} ({config}) is not resolved by the official grader; it cannot be featured as a success")
    issue = issues[instance_id]
    title, _, body = issue["problem_statement"].strip().partition("\n")
    patch_files = [line.split(" b/", 1)[1] for line in result.model_patch.splitlines() if line.startswith("diff --git ")]
    return {
        "id": f"{instance_id}-{config}",
        "instanceId": instance_id,
        "repo": issue["repo"],
        "version": issue.get("version"),
        "tag": tag,
        "takeaway": takeaway,
        "config": {"key": config, **CONFIGS[config]},
        "issue": {"title": title.strip(), "body": body.strip()},
        "retrieval": [{"path": f, "inPatch": f in patch_files} for f in result.retrieved_files],
        "plan": {"files": [{"path": f, "inPatch": f in patch_files} for f in result.planned_files]},
        "attempts": _attempts(result, CONFIGS[config]["judge"]),
        "maxAttempts": MAX_ATTEMPTS,
        "selfAccepted": result.self_accepted,
        "patch": result.model_patch,
        "patchFiles": patch_files,
        "grade": {
            "resolved": result.grade.resolved,
            "failToPass": [result.grade.fail_to_pass_passed, result.grade.fail_to_pass_total],
            "passToPass": [result.grade.pass_to_pass_passed, result.grade.pass_to_pass_total],
        },
        "metrics": {
            "attempts": result.attempts,
            "llmCalls": result.llm_calls,
            "toolCalls": result.tool_calls,
            "toolErrors": result.tool_errors,
            "inputTokens": result.input_tokens,
            "outputTokens": result.output_tokens,
            "cacheReadTokens": result.cache_read_tokens,
            "costUsd": round(result.cost_usd or 0, 4),
            "latencySeconds": round(result.latency_seconds),
        },
        "source": f"evals/results/{folder}/{config}/tasks/{instance_id}.json",
    }


def main():
    from datasets import load_dataset

    wanted = {instance_id for _, _, instance_id, _, _ in FEATURED}
    issues = {r["instance_id"]: r for r in load_dataset(SWEBENCH_LITE, split="test") if r["instance_id"] in wanted}
    replays = [build_replay(*entry, issues) for entry in FEATURED]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"model": "claude-sonnet-5", "benchmark": "SWE-bench Lite", "replays": replays}, indent=2) + "\n")
    print(f"wrote {OUTPUT.relative_to(PROJECT_ROOT)}: {', '.join(r['id'] for r in replays)}")


if __name__ == "__main__":
    main()
