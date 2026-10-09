"""Run PatchPilot over an eval suite.

    python -m evals.run --suite smoke --dry-run --limit 3      # set up environments only; no LLM calls
    python -m evals.run --suite smoke --config full --limit 3  # real run, asks before spending
    python -m evals.run --suite smoke --experiment ablation    # baseline, +rag, +planner, full

Results land in evals/results/<name>/<config>/. Reruns resume: finished tasks are skipped.
"""
import argparse
import asyncio
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from agent.errors import describe_error, is_fatal
from agent.run import run_issue
from evals import results as store
from evals.configs import CONFIGS, EXPERIMENTS
from evals.environments import (
    DEFAULT_SANDBOX_MEMORY,
    SWEBENCH_PLATFORM,
    PreparedTask,
    prepare,
)
from evals.results import Grade, TaskResult
from evals.tasks import EVALS_DIR, Task, load_suite
from sandbox.runner import Sandbox, destroy_run
from tools import git, testing

RESULTS_DIR = EVALS_DIR / "results"


def _result_from_report(task: Task, config: str, report) -> TaskResult:
    state = report.state
    result = TaskResult(
        instance_id=task.instance_id,
        source=task.source,
        config=config,
        status="agent_error" if report.error else "done",
        error=report.error,
        self_accepted=report.success,
        attempts=report.attempts,
        retries=report.retries,
        tool_calls=report.tool_calls,
        tool_errors=report.tool_errors,
        llm_calls=report.llm_calls,
        input_tokens=report.input_tokens,
        output_tokens=report.output_tokens,
        cache_read_tokens=report.cache_read_tokens,
        cache_write_tokens=report.cache_write_tokens,
        max_prompt_tokens=report.max_prompt_tokens,
        mean_prompt_tokens=report.mean_prompt_tokens,
        cost_usd=report.cost_usd,
        latency_seconds=report.latency_seconds,
        tests_before_passed=report.tests_before_passed,
        tests_after_passed=report.tests_after_passed,
        retrieved_files=state.retrieved_files,
        planned_files=state.plan.relevant_files if state.plan else [],
        gold_files=task.gold_files,
        model_patch=state.diff or "",
        trace_id=report.trace_id,
        degradations=report.degradations,
        final_evaluation=state.evaluation.model_dump() if state.evaluation else None,
        rejections=[
            r.model_dump(include={"attempt", "tests_ran", "tests_passed", "tests_selected", "problems"})
            for r in state.failure_history
        ],
    )
    if task.source == "custom":
        # Custom tasks are graded by their own full test suite, which the agent's final check already ran.
        result.grade = Grade(grader="local", resolved=bool(report.tests_after_passed),
                             patch_applied=bool(result.model_patch))
    return result


async def run_task(
    task: Task, config: str, *, llm, model_name: str, retriever, work_dir: Path, max_attempts: int,
    platform: str | None = SWEBENCH_PLATFORM, sandbox_memory: str = DEFAULT_SANDBOX_MEMORY,
) -> TaskResult:
    try:
        # Image pulls and workspace copies block for minutes; a thread keeps the other runs moving.
        prepared = await asyncio.to_thread(prepare, task, work_dir, platform, sandbox_memory)
    except Exception as e:  # noqa: BLE001 - any setup failure is this task's env_error, never the whole eval's
        return TaskResult(instance_id=task.instance_id, source=task.source, config=config, status="env_error",
                          error=f"{type(e).__name__}: {e}", gold_files=task.gold_files)
    state = {"checks": prepared.checks, "max_attempts": max_attempts}
    if prepared.workspace:
        state["prepared_workspace"] = prepared.workspace
    report = await run_issue(
        task.repo_path or task.repo,
        task.problem_statement,
        llm=llm,
        model_name=model_name,
        retriever=retriever,
        sandbox_config=prepared.sandbox,
        workspace_root=str(work_dir),
        agent_config=CONFIGS[config],
        raise_errors=False,
        **state,
    )
    return _result_from_report(task, config, report)


async def run_suite(
    tasks: list[Task],
    configs: list[str],
    out_dir: Path,
    *,
    llm,
    model_name: str,
    retriever,
    max_attempts: int = 3,
    max_cost: float | None = None,
    keep_workspaces: bool = False,
    platform: str | None = SWEBENCH_PLATFORM,
    parallel: int = 1,
    sandbox_memory: str = DEFAULT_SANDBOX_MEMORY,
    log=print,
) -> list[TaskResult]:
    """Run every (config, task) pair not already done, up to `parallel` at a time.

    Pairs start in config order, so a budget stop leaves the earlier configs complete. The budget is checked before
    each run starts; runs already in flight finish, so the total can overshoot by up to parallel - 1 runs.
    """
    spent = 0.0
    done: list[TaskResult] = []
    stopped = False
    gate = asyncio.Semaphore(max(parallel, 1))
    write_lock = asyncio.Lock()

    async def one(config: str, task: Task) -> None:
        nonlocal spent, stopped
        config_dir = out_dir / config
        async with gate:
            if max_cost is not None and spent >= max_cost:
                if not stopped:
                    log(f"Stopping: ${spent:.2f} spent reaches the --max-cost limit of ${max_cost:.2f}.")
                stopped = True
                return
            work_dir = Path(tempfile.mkdtemp(prefix="patchpilot-eval-"))
            try:
                result = await run_task(task, config, llm=llm, model_name=model_name, retriever=retriever,
                                        work_dir=work_dir, max_attempts=max_attempts, platform=platform,
                                        sandbox_memory=sandbox_memory)
            except (Exception, BaseExceptionGroup) as e:
                # Last line of defence: one task's unexpected failure is that task's result, never the batch's.
                if is_fatal(e):
                    raise
                result = TaskResult(instance_id=task.instance_id, source=task.source, config=config,
                                    status="agent_error", gold_files=task.gold_files,
                                    error=f"unexpected crash outside the agent run: {describe_error(e)}")
            finally:
                if not keep_workspaces:
                    await asyncio.to_thread(shutil.rmtree, work_dir, True)
            async with write_lock:
                store.save(config_dir, result)
                store.write_predictions(config_dir)
                spent += result.cost_usd or 0.0
                done.append(result)
            log(
                f"[{config}] {task.instance_id}: {result.status}, accepted={result.self_accepted}, "
                f"attempts={result.attempts}, tokens={result.total_tokens}, cost=${result.cost_usd or 0:.4f}, "
                f"{result.latency_seconds:.0f}s" + (f" ({result.error})" if result.error else "")
            )

    jobs = []
    for config in configs:
        for task in tasks:
            if store.task_path(out_dir / config, task.instance_id).exists():
                log(f"[{config}] {task.instance_id}: already done, skipping")
            else:
                jobs.append(one(config, task))
    await asyncio.gather(*jobs)
    return done


def _gigabytes(size: str) -> float:
    units = {"k": 1 / 1024**2, "m": 1 / 1024, "g": 1.0}
    return float(size[:-1]) * units[size[-1].lower()] if size[-1].lower() in units else float(size) / 1024**3


# Main-process memory: Python, the agent, and the embedding model. Indexing a large repo on CPU peaked at ~4.7 GB
# with batch 16; on a GPU the model's working memory moves to the card.
MAIN_PROCESS_GB = {"cpu": 4.5, "cuda": 2.0}


def memory_plan(parallel: int, sandbox_memory: str, device: str, host_gb: float | None = None) -> tuple[float, float, str]:
    """(planned peak GB, host RAM GB, verdict). Sandboxes are capped by Docker, so their caps are the worst case."""
    if host_gb is None:
        host_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    peak = parallel * _gigabytes(sandbox_memory) + MAIN_PROCESS_GB.get(device, MAIN_PROCESS_GB["cpu"])
    if peak <= host_gb:
        verdict = "fits in RAM"
    else:
        verdict = (f"exceeds RAM by {peak - host_gb:.1f} GB in the worst case (all sandboxes at their cap while "
                   f"indexing); lower --parallel or --sandbox-memory if runs get slow or killed")
    return peak, host_gb, verdict


def dry_run(tasks: list[Task], out_dir: Path, retriever=None, log=print) -> list[dict]:
    """Check each task's environment end to end without the agent: image, workspace, sandbox, test runner, index.

    Indexing follows the agent's path (source up front, then tests related to what a search for the issue finds),
    which also warms the embedding cache. Test selection uses the gold files as a stand-in for the files the agent
    would change; this is only for validating setup, and the agent itself never sees gold files.
    """
    from agent.graph import (
        MAX_SELECTED_FILES,
        RETRIEVE_TOP_K,
        index_related_tests,
        index_source,
        source_files,
    )
    from agent.state import IndexStats
    from retrieval.retriever import chunk_paths, select_files

    rows = []
    # Unique per invocation: two dry runs of the same task (say, from two terminals) must never share a container
    # name or cleanup label, or one run's cleanup removes the other's sandbox mid-task.
    invocation = uuid.uuid4().hex[:8]
    for task in tasks:
        row: dict = {"instance_id": task.instance_id}
        work_dir = Path(tempfile.mkdtemp(prefix="patchpilot-dry-"))
        run_id = f"dry-{invocation}-{task.instance_id.replace('__', '-')[-40:]}"
        try:
            started = time.monotonic()
            prepared: PreparedTask = prepare(task, work_dir)
            repo = Path(prepared.workspace) if prepared.workspace else git.checkout_repo(task.repo_path, work_dir / "repo")
            row["prepare_seconds"] = round(time.monotonic() - started, 1)
            files = git.list_files(repo)
            row["files"] = len(files)

            started = time.monotonic()
            if retriever is not None:
                stats = index_source(retriever, run_id, repo, files)
                found = select_files(retriever.search(run_id, task.problem_statement, top_k=RETRIEVE_TOP_K),
                                     MAX_SELECTED_FILES)
                stats = index_related_tests(retriever, run_id, repo, files, source_files(found), stats)
                retriever.drop(run_id)
                row["index_seconds"] = round(time.monotonic() - started, 1)
            else:
                _, report = chunk_paths(repo, source_files(files))
                stats = IndexStats(source_chunks=report.chunks, skipped_chunks=report.skipped_chunks)
            unindexed = [f for f in files if testing.is_test_module(f) and f not in stats.test_files]
            row["chunks"] = {
                "source": stats.source_chunks,
                "tests": stats.test_chunks,
                "skipped": stats.skipped_chunks,
                "unrelated_tests_not_indexed": chunk_paths(repo, unindexed)[1].chunks,
            }
            row["related_tests_indexed"] = stats.test_files if retriever is not None else "n/a (--skip-index)"

            started = time.monotonic()
            sandbox = Sandbox.start(repo, run_id, prepared.sandbox)
            command = prepared.checks.test_command
            paths = [] if command.run_all_when_unselected else testing.select_tests(repo, files, task.gold_files)
            row["gold_files"] = task.gold_files
            row["tests_selected"] = paths or "full suite"
            result = testing.run_tests(sandbox, command, paths)
            row["test_runner_ok"] = result.ran and result.exit_code is not None and not result.timed_out
            row["test_exit_code"] = result.exit_code
            row["test_output_tail"] = result.output.strip()[-300:]
            row["test_seconds"] = round(time.monotonic() - started, 1)
            row["status"] = "ok"
        except Exception as e:  # noqa: BLE001 - a dry run reports every environment problem rather than stopping
            row["status"] = "error"
            row["error"] = f"{type(e).__name__}: {e}"
        finally:
            destroy_run(run_id)
            shutil.rmtree(work_dir, ignore_errors=True)
        rows.append(row)
        log(json.dumps(row))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dry_run.json").write_text(json.dumps(rows, indent=2))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default="smoke", help="smoke (15 SWE-bench Lite tasks), custom, or lite (all 300)")
    parser.add_argument("--config", choices=sorted(CONFIGS), help="one configuration")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), help="a set of configurations to compare")
    parser.add_argument("--name", default=None, help="results folder name (default: the suite name)")
    parser.add_argument("--limit", type=int, help="only the first N tasks of the suite")
    parser.add_argument("--instance", action="append", help="only these instance ids (repeatable)")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-cost", type=float, default=None, help="stop starting new tasks after this many USD")
    parser.add_argument("--dry-run", action="store_true", help="set up environments and check test runners; no LLM")
    parser.add_argument("--skip-index", action="store_true", help="dry run: skip indexing (and cache warming)")
    parser.add_argument("--keep-workspaces", action="store_true")
    parser.add_argument("--parallel", type=int, default=2, help="agent runs at once (default 2)")
    parser.add_argument("--sandbox-memory", default=DEFAULT_SANDBOX_MEMORY,
                        help=f"memory cap per SWE-bench sandbox (default {DEFAULT_SANDBOX_MEMORY})")
    parser.add_argument("--batch", action="store_true",
                        help="send LLM calls through the Anthropic Batch API: half the price, minutes of latency per call")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation before spending")
    args = parser.parse_args()

    tasks = load_suite(args.suite)
    if args.instance:
        tasks = [t for t in tasks if t.instance_id in set(args.instance)]
    if args.limit:
        tasks = tasks[: args.limit]
    out_dir = RESULTS_DIR / (args.name or args.suite)

    from retrieval.embeddings import load_embedder
    from retrieval.retriever import Retriever
    from retrieval.store import ChunkStore, database_url

    def make_retriever():
        embedder = load_embedder()
        print(f"Embeddings: {embedder.describe()}", flush=True)

        def progress(done, total):
            print(f"  embedding new chunks: {done}/{total}", flush=True)  # flush: progress must show when piped

        return Retriever(ChunkStore(database_url(), embedder.dim), embedder, progress)

    if args.dry_run:
        dry_run(tasks, out_dir, None if args.skip_index else make_retriever(), log=lambda line: print(line, flush=True))
        return

    if bool(args.config) == bool(args.experiment):
        parser.error("pass exactly one of --config or --experiment (or --dry-run)")
    configs = EXPERIMENTS[args.experiment] if args.experiment else [args.config]

    from agent.llm import get_chat_model, load_llm_settings

    settings = load_llm_settings()
    pending = [(c, t) for c in configs for t in tasks if not store.task_path(out_dir / c, t.instance_id).exists()]
    runs = len(tasks) * len(configs)
    print(f"About to run {len(pending)} of {runs} agent run(s) ({len(tasks)} task(s) x {len(configs)} config(s); "
          f"the rest are already done) with {settings.provider}/{settings.model}, up to {args.max_attempts} attempts each.")
    print(f"Configs: {', '.join(configs)}. Results: {out_dir}")
    print(f"LLM calls: {'Batch API (half price; each call waits for its batch, usually minutes)' if args.batch else 'standard API'}")
    print(f"Spending cap: {'$' + format(args.max_cost, '.2f') if args.max_cost is not None else 'none (--max-cost)'}")
    device = (os.getenv("EMBEDDING_DEVICE") or "cpu").strip().lower()
    peak, host, verdict = memory_plan(args.parallel, args.sandbox_memory, device)
    print(f"Memory: {args.parallel} parallel x {args.sandbox_memory} sandboxes + main process ({device} embeddings) "
          f"= up to {peak:.1f} GB of {host:.1f} GB RAM: {verdict}")
    if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        print("Cancelled.")
        return

    results = asyncio.run(run_suite(
        tasks, configs, out_dir, llm=get_chat_model(settings, batch=args.batch), model_name=settings.model,
        retriever=make_retriever(), max_attempts=args.max_attempts, max_cost=args.max_cost,
        keep_workspaces=args.keep_workspaces, parallel=args.parallel, sandbox_memory=args.sandbox_memory,
    ))
    spent = sum(r.cost_usd or 0 for r in results)
    print(f"\nFinished {len(results)} run(s), ${spent:.2f}. Next: python -m evals.grade {out_dir}")


if __name__ == "__main__":
    main()
