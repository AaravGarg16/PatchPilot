"""Grade a finished run with the official SWE-bench harness, then merge its verdicts into our results.

    python -m evals.grade evals/results/smoke            # every config in the run
    python -m evals.grade evals/results/smoke/full       # one config

Grading pulls the same task images and runs the hidden tests. Custom tasks are already graded locally.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from evals import results as store
from evals.results import Grade

# The harness writes logs under <cwd>/logs/run_evaluation/<run_id>/<model>/<instance>/report.json.
LOG_DIR = Path("logs") / "run_evaluation"


def run_id_for(config_dir: Path) -> str:
    return f"{config_dir.parent.name}-{config_dir.name}"


def read_grade(config_dir: Path, run_id: str, config: str, instance_id: str, patch: str) -> Grade:
    if not patch.strip():
        return Grade(grader="swebench", resolved=False, patch_applied=False, error="empty patch")
    report_file = config_dir / LOG_DIR / run_id / store.model_name(config) / instance_id / "report.json"
    if not report_file.exists():
        return Grade(grader="swebench", resolved=None, patch_applied=None, error="no report from the harness")
    report = json.loads(report_file.read_text()).get(instance_id, {})
    tests = report.get("tests_status", {})
    f2p, p2p = tests.get("FAIL_TO_PASS", {}), tests.get("PASS_TO_PASS", {})
    return Grade(
        grader="swebench",
        resolved=bool(report.get("resolved")),
        patch_applied=bool(report.get("patch_successfully_applied")),
        fail_to_pass_passed=len(f2p.get("success", [])),
        fail_to_pass_total=len(f2p.get("success", [])) + len(f2p.get("failure", [])),
        pass_to_pass_passed=len(p2p.get("success", [])),
        pass_to_pass_total=len(p2p.get("success", [])) + len(p2p.get("failure", [])),
    )


def harness_command(predictions: Path, dataset: str, run_id: str, instance_ids: list[str], workers: int,
                    timeout: int) -> list[str]:
    return [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset,
        "--split", "test",
        "--predictions_path", str(predictions.resolve()),
        "--run_id", run_id,
        "--max_workers", str(workers),
        "--timeout", str(timeout),
        "--report_dir", ".",
        "--instance_ids", *instance_ids,
    ]


def grade_config(config_dir: Path, dataset: str, workers: int = 1, timeout: int = 1800, run_harness: bool = True) -> None:
    results = store.load_all(config_dir)
    to_grade = [r for r in results if r.source == "swebench" and r.status != "env_error"]
    if not to_grade:
        print(f"{config_dir}: nothing to grade with SWE-bench")
        return
    run_id = run_id_for(config_dir)
    with_patches = [r.instance_id for r in to_grade if r.model_patch.strip()]
    if run_harness and with_patches:
        predictions = store.write_predictions(config_dir)
        command = harness_command(predictions, dataset, run_id, with_patches, workers, timeout)
        print(f"{config_dir}: grading {len(with_patches)} patch(es) with the SWE-bench harness...")
        subprocess.run(command, cwd=config_dir, check=False)
    for result in to_grade:
        result.grade = read_grade(config_dir, run_id, result.config, result.instance_id, result.model_patch)
        store.save(config_dir, result)
    resolved = sum(bool(r.grade.resolved) for r in to_grade)
    print(f"{config_dir}: {resolved}/{len(to_grade)} resolved")


def config_dirs(path: Path) -> list[Path]:
    path = Path(path)
    if (path / "tasks").is_dir():
        return [path]
    return sorted(p for p in path.iterdir() if (p / "tasks").is_dir())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="a run folder or one config folder inside it")
    parser.add_argument("--dataset", default="SWE-bench/SWE-bench_Lite")
    parser.add_argument("--workers", type=int, default=1, help="parallel containers; emulated x86 is heavy")
    parser.add_argument("--timeout", type=int, default=1800, help="per-instance seconds")
    parser.add_argument("--merge-only", action="store_true", help="re-read existing harness reports; run nothing")
    args = parser.parse_args()
    for config_dir in config_dirs(args.path):
        grade_config(config_dir, args.dataset, args.workers, args.timeout, run_harness=not args.merge_only)
    print(f"Next: python -m evals.report {args.path}")


if __name__ == "__main__":
    main()
