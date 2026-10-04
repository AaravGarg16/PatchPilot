"""Eval tasks: SWE-bench Lite instances and our own custom tasks, in one shape."""
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

EVALS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALS_DIR.parent
SWEBENCH_LITE = "SWE-bench/SWE-bench_Lite"


class Task(BaseModel):
    instance_id: str
    source: Literal["swebench", "custom"]
    repo: str
    problem_statement: str
    # Scoring only (retrieval precision). Never put into the agent's state or prompts.
    gold_files: list[str] = []

    # SWE-bench
    dataset: str | None = None
    base_commit: str | None = None
    image: str | None = None
    version: str | None = None

    # Custom: a local directory, cloned like any repo; graded by its own test suite.
    repo_path: str | None = None


def files_in_patch(patch: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"^diff --git a/(\S+) b/", patch, re.MULTILINE)))


def task_from_swebench_row(row: dict, dataset: str = SWEBENCH_LITE) -> Task:
    return Task(
        instance_id=row["instance_id"],
        source="swebench",
        repo=row["repo"],
        problem_statement=row["problem_statement"],
        gold_files=files_in_patch(row["patch"]),
        dataset=dataset,
        base_commit=row["base_commit"],
        image=row["image"],
        version=row.get("version"),
    )


def load_swebench(instance_ids: list[str] | None, dataset: str = SWEBENCH_LITE, split: str = "test") -> list[Task]:
    """Downloads the dataset's metadata (a few MB, cached by HuggingFace) on first use."""
    from datasets import load_dataset

    rows = {row["instance_id"]: row for row in load_dataset(dataset, split=split)}
    ids = instance_ids or sorted(rows)
    missing = [i for i in ids if i not in rows]
    if missing:
        raise ValueError(f"not in {dataset}: {', '.join(missing)}")
    return [task_from_swebench_row(rows[i], dataset) for i in ids]


def load_custom(path: Path) -> list[Task]:
    tasks = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            record["repo_path"] = str((PROJECT_ROOT / record["repo_path"]).resolve())
            tasks.append(Task(source="custom", **record))
    return tasks


def load_suite(name: str) -> list[Task]:
    """smoke: the curated SWE-bench Lite subset. custom: our own tasks. lite: all 300 Lite tasks."""
    if name == "custom":
        return load_custom(EVALS_DIR / "tasks" / "custom.jsonl")
    if name == "smoke":
        spec = json.loads((EVALS_DIR / "tasks" / "swebench_lite_smoke.json").read_text())
        return load_swebench(spec["instance_ids"], spec["dataset"])
    if name == "lite":
        return load_swebench(None)
    raise ValueError(f"unknown suite {name!r}; use smoke, custom or lite")
