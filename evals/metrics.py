"""The metric list from the spec, computed from per-task results."""
import math
from statistics import mean

from pydantic import BaseModel

from evals.results import TaskResult

# Retrieved files are the top files picked for the prompt (MAX_SELECTED_FILES in agent/graph.py).


class Summary(BaseModel):
    config: str
    tasks: int  # attempted, excluding environment failures
    env_errors: int
    agent_errors: int
    fallback_runs: int  # runs where a step's output was replaced by a fallback (plan, verdict, or an ended attempt)
    graded: int  # tasks with a verdict
    success_rate: float | None  # resolved / graded
    tests_passed: float | None  # share of the grader's target tests (FAIL_TO_PASS) that pass
    patch_acceptance_rate: float | None  # non-empty patch that applies cleanly / graded
    self_acceptance_rate: float  # the agent accepted its own patch
    self_acceptance_precision: float | None  # of those it accepted, share actually resolved
    invalid_tool_calls_per_task: float
    avg_retries: float
    input_tokens_per_task: float
    output_tokens_per_task: float
    tokens_per_task: float
    tokens_per_successful_task: float | None  # all tokens spent / tasks resolved
    cost_per_task: float | None
    p50_latency: float
    p95_latency: float
    retrieval_precision: float | None  # share of retrieved files that the gold patch touches
    retrieval_recall: float | None  # share of gold-patch files that were retrieved
    cached_input_share: float | None  # share of input tokens served from the prompt cache
    mean_prompt_tokens: float  # context utilization: average prompt size per LLM call
    max_prompt_tokens: int


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile: always an observed value, which is honest for small task counts."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(math.ceil(q / 100 * len(ordered)) - 1, 0)]


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _overlap(retrieved: list[str], gold: list[str]) -> tuple[float, float] | None:
    if not retrieved or not gold:
        return None
    hits = len(set(retrieved) & set(gold))
    return hits / len(set(retrieved)), hits / len(set(gold))


def summarize(config: str, results: list[TaskResult]) -> Summary:
    attempted = [r for r in results if r.status != "env_error"]
    graded = [r for r in attempted if r.grade and r.grade.resolved is not None]
    resolved = [r for r in graded if r.grade.resolved]
    accepted = [r for r in attempted if r.self_accepted]
    accepted_graded = [r for r in accepted if r.grade and r.grade.resolved is not None]
    f2p = [r.grade.fail_to_pass_passed / r.grade.fail_to_pass_total for r in graded if r.grade.fail_to_pass_total]
    costs = [r.cost_usd for r in attempted]
    overlaps = [o for r in attempted if (o := _overlap(r.retrieved_files, r.gold_files))]
    n = len(attempted) or 1
    total_input = sum(r.input_tokens for r in attempted)

    return Summary(
        config=config,
        tasks=len(attempted),
        env_errors=len(results) - len(attempted),
        agent_errors=sum(r.status == "agent_error" for r in attempted),
        fallback_runs=sum(any(note.startswith("fallback:") for note in r.degradations) for r in attempted),
        graded=len(graded),
        success_rate=len(resolved) / len(graded) if graded else None,
        tests_passed=_mean(f2p),
        patch_acceptance_rate=_mean([float(bool(r.grade.patch_applied)) for r in graded]),
        self_acceptance_rate=len(accepted) / n,
        self_acceptance_precision=(
            sum(bool(r.grade.resolved) for r in accepted_graded) / len(accepted_graded) if accepted_graded else None
        ),
        invalid_tool_calls_per_task=sum(r.tool_errors for r in attempted) / n,
        avg_retries=sum(r.retries for r in attempted) / n,
        input_tokens_per_task=sum(r.input_tokens for r in attempted) / n,
        output_tokens_per_task=sum(r.output_tokens for r in attempted) / n,
        tokens_per_task=sum(r.total_tokens for r in attempted) / n,
        tokens_per_successful_task=sum(r.total_tokens for r in attempted) / len(resolved) if resolved else None,
        cost_per_task=None if not costs or None in costs else sum(costs) / n,
        p50_latency=percentile([r.latency_seconds for r in attempted], 50),
        p95_latency=percentile([r.latency_seconds for r in attempted], 95),
        retrieval_precision=_mean([p for p, _ in overlaps]),
        retrieval_recall=_mean([r for _, r in overlaps]),
        cached_input_share=(sum(r.cache_read_tokens for r in attempted) / total_input) if total_input else None,
        mean_prompt_tokens=_mean([r.mean_prompt_tokens for r in attempted if r.llm_calls]) or 0.0,
        max_prompt_tokens=max((r.max_prompt_tokens for r in attempted), default=0),
    )
