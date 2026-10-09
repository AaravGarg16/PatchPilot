"""Parallel runs, memory planning, suite-filtered reports and the cost estimate. No Docker, no LLM."""
import asyncio

import evals.run as run_module
from evals import results as store
from evals.estimate import ASSUMED, estimate
from evals.report import build_report
from evals.results import Grade, TaskResult
from evals.run import memory_plan, run_suite
from evals.tasks import Task, suite_instance_ids


def fake_task(i: int) -> Task:
    return Task(instance_id=f"t__t-{i}", source="custom", repo="t/t", problem_statement="x", repo_path=".")


def test_runs_execute_in_parallel_up_to_the_limit_and_in_config_order(monkeypatch, tmp_path):
    active, peak, order = 0, 0, []

    async def fake_run_task(task, config, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        order.append((config, task.instance_id))
        await asyncio.sleep(0.02)
        active -= 1
        return TaskResult(instance_id=task.instance_id, source="custom", config=config, status="done", cost_usd=0.1)

    monkeypatch.setattr(run_module, "run_task", fake_run_task)
    tasks = [fake_task(i) for i in range(4)]
    done = asyncio.run(run_suite(tasks, ["baseline", "full"], tmp_path, llm=None, model_name="m", retriever=None,
                                 parallel=2, log=lambda _: None))
    assert len(done) == 8 and peak == 2
    assert [c for c, _ in order[:4]] == ["baseline"] * 4  # configs start in order
    assert len(store.load_all(tmp_path / "full")) == 4


def test_cost_cap_stops_new_runs(monkeypatch, tmp_path):
    async def fake_run_task(task, config, **kwargs):
        return TaskResult(instance_id=task.instance_id, source="custom", config=config, status="done", cost_usd=1.0)

    monkeypatch.setattr(run_module, "run_task", fake_run_task)
    logs: list[str] = []
    done = asyncio.run(run_suite([fake_task(i) for i in range(5)], ["full"], tmp_path, llm=None, model_name="m",
                                 retriever=None, parallel=1, max_cost=2.0, log=logs.append))
    assert len(done) == 2 and any("--max-cost" in line for line in logs)


def test_memory_plan():
    peak, host, verdict = memory_plan(2, "2g", "cuda", host_gb=8.0)
    assert (peak, host, verdict) == (6.0, 8.0, "fits in RAM")
    peak, _, verdict = memory_plan(2, "4g", "cpu", host_gb=8.0)
    assert peak == 12.5 and verdict.startswith("exceeds RAM by 4.5 GB")


def test_report_can_be_limited_to_one_suite(tmp_path):
    ablation = suite_instance_ids("ablation8")
    extra = next(i for i in suite_instance_ids("lite20") if i not in ablation)
    for instance_id in [*ablation, extra]:
        store.save(tmp_path / "full", TaskResult(
            instance_id=instance_id, source="swebench", config="full", status="done", llm_calls=1,
            grade=Grade(grader="swebench", resolved=instance_id == extra, patch_applied=True)))
    assert "| Tasks (graded / attempted) | 8/8 |" in build_report(tmp_path, "ablation8")
    assert "| Tasks (graded / attempted) | 9/9 |" in build_report(tmp_path)


def test_estimate_counts_shared_runs_once():
    rows = {r["experiment"]: r for r in estimate("claude-sonnet-5")}
    assert (rows["Component ablation"]["runs"], rows["Component ablation"]["new_runs"]) == (32, 32)
    assert rows["Headline (full agent)"]["new_runs"] == 12  # 8 of lite20's 20 come from the ablation's full runs
    assert rows["Context engineering"]["new_runs"] == 8  # full is reused; only full-history is new
    assert all(0 < r["cached"] < r["standard"] for r in rows.values())


def test_estimate_uses_measurements_and_skips_finished_runs():
    first = suite_instance_ids("ablation8")[0]
    rows = estimate("claude-sonnet-5", measured={"baseline": (100_000, 1_000)}, done={("baseline", first)})
    ablation = rows[0]
    assert ablation["new_runs"] == 31
    cheaper = estimate("claude-sonnet-5", measured={"baseline": (100_000, 1_000)})[0]["cached"]
    assumed = estimate("claude-sonnet-5")[0]["cached"]
    assert cheaper < assumed and ASSUMED["baseline"][0] > 100_000
