import asyncio
import json
import subprocess

import pytest

from agent.schemas import Evaluation, Plan
from evals import results as store
from evals.configs import CONFIGS, EXPERIMENTS
from evals.environments import (
    freeze_baseline,
    neutralize_git_config,
    prepare,
)
from evals.grade import harness_command, read_grade, run_id_for
from evals.metrics import percentile, summarize
from evals.report import build_report
from evals.results import Grade, TaskResult
from evals.run import run_suite
from evals.tasks import (
    Task,
    files_in_patch,
    load_custom,
    load_suite,
    task_from_swebench_row,
)
from sandbox.runner import Sandbox
from tests.fakes import (
    FIX_PAGINATE,
    FIX_TOTAL_PAGES,
    FIXTURE_REPO,
    FakeLLM,
    done,
    tool_call,
)
from tests.test_graph import FINISH, PLAN
from tools import filesystem, git, testing

GOLD_PATCH = """diff --git a/shop/pagination.py b/shop/pagination.py
--- a/shop/pagination.py
+++ b/shop/pagination.py
@@ -1 +1 @@
-x
+y
diff --git a/shop/extra.py b/shop/extra.py
"""


def swebench_row(**overrides) -> dict:
    row = {
        "instance_id": "owner__repo-1", "repo": "owner/repo", "problem_statement": "It breaks.",
        "patch": GOLD_PATCH, "test_patch": "diff --git a/tests/test_hidden.py b/tests/test_hidden.py\n",
        "base_commit": "abc123", "image": "swebench/sweb.eval.x86_64.owner_1776_repo-1:latest", "version": "1.0",
        "FAIL_TO_PASS": ["tests/test_hidden.py::test_it"], "PASS_TO_PASS": [],
    }
    return {**row, **overrides}


# --- tasks ---------------------------------------------------------------------------------------------------------


def test_gold_files_come_from_the_patch_and_hidden_tests_never_enter_the_task():
    task = task_from_swebench_row(swebench_row())
    assert task.gold_files == ["shop/pagination.py", "shop/extra.py"]
    dumped = task.model_dump_json()
    assert "test_hidden" not in dumped and "FAIL_TO_PASS" not in dumped
    assert files_in_patch("") == []


def test_suites_load():
    (custom,) = load_custom(__import__("evals.tasks").tasks.EVALS_DIR / "tasks" / "custom.jsonl")
    assert custom.source == "custom" and custom.repo_path.endswith("examples/pagination_repo")
    with pytest.raises(ValueError, match="unknown suite"):
        load_suite("nope")


def test_smoke_subset_is_small_and_spans_runners():
    spec = json.loads((__import__("evals.tasks").tasks.EVALS_DIR / "tasks" / "swebench_lite_smoke.json").read_text())
    ids = spec["instance_ids"]
    assert 10 <= len(ids) <= 20 and len(set(ids)) == len(ids)
    assert any(i.startswith("django__") for i in ids) and any(i.startswith("sympy__") for i in ids)


def test_configs_cover_both_experiments():
    assert EXPERIMENTS["ablation"] == ["baseline", "rag", "planner", "full"]
    assert CONFIGS["baseline"].rag is CONFIGS["baseline"].planner is CONFIGS["baseline"].evaluator is False
    assert CONFIGS["full-history"].context_mode == "full_history"


# --- metrics and report --------------------------------------------------------------------------------------------


def result(config="full", resolved=True, tokens=(1000, 100), retries=0, latency=10.0, status="done", **kw) -> TaskResult:
    grade = None if status == "env_error" else Grade(
        grader="swebench", resolved=resolved, patch_applied=True, fail_to_pass_passed=int(resolved), fail_to_pass_total=1
    )
    return TaskResult(instance_id=kw.pop("instance_id", f"t{latency}"), source="swebench", config=config,
                      status=status, input_tokens=tokens[0], output_tokens=tokens[1], retries=retries,
                      latency_seconds=latency, cost_usd=0.01, llm_calls=2, mean_prompt_tokens=tokens[0] / 2,
                      max_prompt_tokens=tokens[0], grade=grade, **kw)


def test_percentile_is_nearest_rank():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
    assert (percentile(values, 50), percentile(values, 95)) == (5, 100)
    assert percentile([], 50) == 0.0


def test_summary_metrics():
    results = [
        result(resolved=True, tokens=(1000, 100), retries=0, latency=10, retrieved_files=["a.py", "b.py"], gold_files=["a.py"]),
        result(resolved=False, tokens=(3000, 300), retries=2, latency=30, self_accepted=True),
        result(status="env_error", latency=1),
    ]
    s = summarize("full", results)
    assert (s.tasks, s.env_errors, s.graded) == (2, 1, 2)
    assert s.success_rate == 0.5
    assert s.tokens_per_task == (1100 + 3300) / 2
    assert s.tokens_per_successful_task == 4400  # everything spent, over one success
    assert s.avg_retries == 1.0
    assert (s.p50_latency, s.p95_latency) == (10, 30)
    assert (s.retrieval_precision, s.retrieval_recall) == (0.5, 1.0)
    assert s.self_acceptance_precision == 0.0  # it accepted the one it got wrong
    assert s.cost_per_task == pytest.approx(0.01)


def test_report_tables(tmp_path):
    for config, tokens in [("baseline", 9000), ("full", 2000), ("full-history", 8000)]:
        store.save(tmp_path / config, result(config=config, tokens=(tokens, 100), instance_id="t1"))
    report = build_report(tmp_path)
    assert "## Component ablation" in report and "Baseline (executor + tests)" in report
    assert "75% fewer input tokens per task than full history, at equal task success" in report
    assert "| Retrieval precision (files) |" in report


# --- grading -------------------------------------------------------------------------------------------------------


def test_grades_are_read_from_the_harness_report(tmp_path):
    config_dir = tmp_path / "smoke" / "full"
    run_id = run_id_for(config_dir)
    report_dir = config_dir / "logs" / "run_evaluation" / run_id / "patchpilot-full" / "owner__repo-1"
    report_dir.mkdir(parents=True)
    (report_dir / "report.json").write_text(json.dumps({"owner__repo-1": {
        "patch_successfully_applied": True, "resolved": False,
        "tests_status": {"FAIL_TO_PASS": {"success": ["a"], "failure": ["b"]}, "PASS_TO_PASS": {"success": ["c"], "failure": []}},
    }}))

    grade = read_grade(config_dir, run_id, "full", "owner__repo-1", "diff --git a/x b/x")
    assert (grade.resolved, grade.patch_applied, grade.fail_to_pass_passed, grade.fail_to_pass_total) == (False, True, 1, 2)
    assert read_grade(config_dir, run_id, "full", "owner__repo-1", "").error == "empty patch"
    assert read_grade(config_dir, run_id, "full", "other", "diff").resolved is None


def test_harness_command_and_predictions(tmp_path):
    store.save(tmp_path, result(instance_id="a__b-1", model_patch="diff --git a/x b/x\n"))
    store.save(tmp_path, result(instance_id="c__d-2", status="env_error"))
    rows = [json.loads(line) for line in store.write_predictions(tmp_path).read_text().splitlines()]
    assert rows == [{"instance_id": "a__b-1", "model_name_or_path": "patchpilot-full", "model_patch": "diff --git a/x b/x\n"}]
    command = harness_command(tmp_path / "predictions.jsonl", "SWE-bench/SWE-bench_Lite", "r", ["a__b-1"], 1, 900)
    assert command[1:3] == ["-m", "swebench.harness.run_evaluation"]
    assert command[command.index("--instance_ids") + 1:] == ["a__b-1"]


# --- running suites (custom task, fake LLM) ------------------------------------------------------------------------


@pytest.fixture
def custom_task() -> Task:
    (task,) = load_suite("custom")
    return task


def full_llm():
    return FakeLLM({Plan: [PLAN], Evaluation: [FINISH]}, executor_turns=[FIX_PAGINATE, FIX_TOTAL_PAGES, done()])


def test_run_suite_records_results_and_resumes(custom_task, retriever, sandbox_image, tmp_path):
    out = tmp_path / "run"
    logs: list[str] = []
    asyncio.run(run_suite([custom_task], ["full"], out, llm=full_llm(), model_name="claude-sonnet-5",
                          retriever=retriever, log=logs.append))

    (saved,) = store.load_all(out / "full")
    assert saved.status == "done" and saved.self_accepted
    assert saved.grade.grader == "local" and saved.grade.resolved
    assert "+    return -(-total_items // per_page)" in saved.model_patch
    assert saved.retrieved_files[0] == "shop/pagination.py" and saved.gold_files == ["shop/pagination.py"]
    assert saved.input_tokens > 0 and saved.cost_usd > 0

    asyncio.run(run_suite([custom_task], ["full"], out, llm=FakeLLM({}), model_name="m", retriever=retriever,
                          log=logs.append))
    assert any("already done, skipping" in line for line in logs)


def test_run_suite_stops_at_the_cost_cap(custom_task, retriever, sandbox_image, tmp_path):
    logs: list[str] = []
    asyncio.run(run_suite([custom_task], ["full", "baseline"], tmp_path, llm=full_llm(), model_name="m",
                          retriever=retriever, max_cost=0.0, log=logs.append))
    assert store.load_all(tmp_path / "full") == []
    assert "--max-cost" in logs[-1]


def test_agent_crash_is_recorded_with_its_cost(custom_task, retriever, sandbox_image, tmp_path):
    bad_plan = PLAN.model_copy(update={"relevant_files": ["nope.py"]})
    llm = FakeLLM({Plan: [bad_plan], Evaluation: [FINISH]})
    asyncio.run(run_suite([custom_task], ["full"], tmp_path, llm=llm, model_name="m", retriever=retriever,
                          log=lambda _: None))
    (saved,) = store.load_all(tmp_path / "full")
    assert saved.status == "agent_error" and "no files that exist" in saved.error
    assert saved.llm_calls == 1 and saved.input_tokens > 0  # the planner call still counts


# --- SWE-bench environment path, with a local look-alike image -----------------------------------------------------


@pytest.fixture(scope="session")
def swebench_like_image(sandbox_image, tmp_path_factory) -> tuple[str, str]:
    """Laid out like a SWE-bench image: the repo at /testbed, Python at the conda env path, and HEAD one commit past
    the task's base commit, as in the published images (a mode-only "SWE-bench" commit). Returns (image, base commit).
    """
    context = tmp_path_factory.mktemp("swebench-like")
    testbed = git.checkout_repo(FIXTURE_REPO, context / "testbed")
    base_commit = git._git(testbed, "rev-parse", "HEAD").strip()
    for tracked in git.list_files(testbed):
        (testbed / tracked).chmod(0o755)
    git._git(testbed, "commit", "--quiet", "--all", "--message", "SWE-bench")
    (testbed / "build_artifact.so").write_text("compiled in the image, untracked\n")
    (context / "Dockerfile").write_text(
        f"FROM {sandbox_image}\n"
        "USER root\n"
        "RUN mkdir -p /opt/miniconda3/envs/testbed/bin && ln -s /usr/local/bin/python /opt/miniconda3/envs/testbed/bin/python\n"
        "COPY testbed /testbed\n"
    )
    tag = "patchpilot-test-swebench-like:latest"
    subprocess.run(["docker", "build", "-q", "-t", tag, str(context)], check=True, capture_output=True)
    return tag, base_commit


def fake_swebench_task(image_and_base: tuple[str, str], base_commit: str | None = None) -> Task:
    image, base = image_and_base
    return Task(instance_id="patchpilot__pagination-swe", source="swebench", repo="patchpilot/pagination",
                problem_statement="Pagination skips the first page", gold_files=["shop/pagination.py"],
                image=image, base_commit=base_commit or base)


def test_prepare_copies_testbed_and_freezes_the_baseline(swebench_like_image, tmp_path):
    _, base_commit = swebench_like_image
    prepared = prepare(fake_swebench_task(swebench_like_image), tmp_path, platform=None)
    repo = git.Path(prepared.workspace)

    assert (repo / "shop" / "pagination.py").exists()
    # The image's extra commit is undone: the workspace is exactly the task's base commit.
    assert git._git(repo, "rev-parse", "HEAD").strip() == base_commit
    assert git.diff(repo) == ""  # the image's untracked build artifact is excluded, not part of the patch
    assert (repo / "build_artifact.so").exists()  # ...but still there, since the environment may need it
    assert "[remote" not in (repo / ".git" / "config").read_text()
    assert prepared.sandbox.workdir == "/testbed" and not prepared.sandbox.install_dependencies
    assert prepared.checks.lint is False and not prepared.checks.test_command.run_all_when_unselected

    # The agent's sandbox runs selected tests against the workspace mounted at /testbed, via the env's PATH.
    box = Sandbox.start(repo, "swe-like", prepared.sandbox)
    try:
        paths = testing.select_tests(repo, git.list_files(repo), ["shop/pagination.py"])
        before = testing.run_tests(box, prepared.checks.test_command, paths)
        not_selected = testing.run_tests(box, prepared.checks.test_command, [])
        cwd = box.exec(["pwd"]).output.strip()
    finally:
        box.destroy()
    assert paths == ["tests/test_pagination.py"]
    assert before.ran and not before.passed and "3 failed" in before.output
    assert not not_selected.ran
    assert cwd == "/testbed"


def test_a_base_commit_missing_from_the_image_is_an_environment_error(swebench_like_image, tmp_path):
    task = fake_swebench_task(swebench_like_image, base_commit="0" * 40)
    with pytest.raises(RuntimeError, match="is not in the image's repository"):
        prepare(task, tmp_path, platform=None)


def test_baseline_checks_refuse_a_wrong_or_dirty_workspace(tmp_path):
    repo = git.checkout_repo(FIXTURE_REPO, tmp_path / "repo")
    with pytest.raises(RuntimeError, match="expected base commit"):
        freeze_baseline(repo, "0" * 40)
    filesystem.write_file(repo, "README.md", "changed\n")
    with pytest.raises(RuntimeError, match="tracked changes"):
        freeze_baseline(repo, None)


def test_neutralized_git_config_drops_remotes_and_commands(tmp_path):
    repo = git.checkout_repo(FIXTURE_REPO, tmp_path / "repo")
    (repo / ".git" / "config").write_text('[core]\n\tfsmonitor = "touch /tmp/pwned"\n[remote "origin"]\n\turl = x\n')
    neutralize_git_config(repo)
    config = (repo / ".git" / "config").read_text()
    assert "fsmonitor" not in config and "remote" not in config
    assert git.diff(repo) == ""


def test_swebench_task_end_to_end_with_fake_llm(swebench_like_image, retriever, tmp_path):
    task = fake_swebench_task(swebench_like_image)
    llm = FakeLLM(
        {Plan: [PLAN], Evaluation: [FINISH]},
        executor_turns=[FIX_PAGINATE, FIX_TOTAL_PAGES, tool_call("run_tests"), done()],
    )
    asyncio.run(run_suite([task], ["full"], tmp_path, llm=llm, model_name="claude-sonnet-5", retriever=retriever,
                          platform=None, log=lambda _: None))

    (saved,) = store.load_all(tmp_path / "full")
    assert saved.status == "done" and saved.self_accepted
    assert saved.grade is None  # SWE-bench tasks wait for the official grader
    assert "build_artifact.so" not in saved.model_patch and "shop/pagination.py" in saved.model_patch
    assert "old mode" not in saved.model_patch  # relative to the base commit, not the image's mode-flipped HEAD
    (prediction,) = [json.loads(line) for line in (tmp_path / "full" / "predictions.jsonl").read_text().splitlines()]
    assert prediction["model_patch"] == saved.model_patch
    # The executor's run_tests tool used the repo's selected tests, not the whole suite.
    tool_output = str(llm.executor_calls[-1][-1].content)
    assert "PASSED (tests/test_pagination.py)" in tool_output


def test_concurrent_dry_runs_never_share_a_container(custom_task, sandbox_image, monkeypatch, tmp_path):
    import evals.run as run_module

    started: list[str] = []
    real_start = run_module.Sandbox.start

    def recording_start(repo, run_id, config=None):
        started.append(run_id)
        return real_start(repo, run_id, config)

    monkeypatch.setattr(run_module.Sandbox, "start", staticmethod(recording_start))
    for _ in range(2):
        (row,) = run_module.dry_run([custom_task], tmp_path, retriever=None, log=lambda _: None)
        assert row["status"] == "ok" and row["test_runner_ok"]
    assert len(started) == 2 and started[0] != started[1]
