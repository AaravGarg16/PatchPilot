import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class Grade(BaseModel):
    grader: Literal["swebench", "local"]
    resolved: bool | None  # None: the grader could not produce a verdict
    patch_applied: bool | None
    fail_to_pass_passed: int = 0
    fail_to_pass_total: int = 0
    pass_to_pass_passed: int = 0
    pass_to_pass_total: int = 0
    error: str | None = None


class TaskResult(BaseModel):
    instance_id: str
    source: Literal["swebench", "custom"]
    config: str
    # env_error: the task's environment could not be set up; excluded from metrics, reported separately.
    status: Literal["done", "agent_error", "env_error"]
    error: str | None = None

    self_accepted: bool = False
    attempts: int = 0
    retries: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0  # part of input_tokens
    cache_write_tokens: int = 0  # part of input_tokens
    max_prompt_tokens: int = 0
    mean_prompt_tokens: float = 0.0
    cost_usd: float | None = None
    latency_seconds: float = 0.0
    tests_before_passed: bool | None = None
    tests_after_passed: bool | None = None

    retrieved_files: list[str] = []
    planned_files: list[str] = []
    gold_files: list[str] = []

    model_patch: str = ""
    trace_id: str | None = None
    # For diagnosing the evaluator: its last verdict, and why each rejected attempt was rejected.
    # Retries and fallbacks after malformed model output; a run with a fallback did not run the normal pipeline.
    degradations: list[str] = []
    final_evaluation: dict | None = None
    rejections: list[dict] = []
    grade: Grade | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def task_path(config_dir: Path, instance_id: str) -> Path:
    return Path(config_dir) / "tasks" / f"{instance_id}.json"


def save(config_dir: Path, result: TaskResult) -> None:
    path = task_path(config_dir, result.instance_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2))


def load_all(config_dir: Path) -> list[TaskResult]:
    return [TaskResult.model_validate_json(p.read_text()) for p in sorted((Path(config_dir) / "tasks").glob("*.json"))]


def model_name(config: str) -> str:
    return f"patchpilot-{config}"


def write_predictions(config_dir: Path) -> Path:
    """SWE-bench's predictions format. Environment failures are left out: there is no patch to grade."""
    path = Path(config_dir) / "predictions.jsonl"
    rows = [
        {"instance_id": r.instance_id, "model_name_or_path": model_name(r.config), "model_patch": r.model_patch}
        for r in load_all(config_dir)
        if r.source == "swebench" and r.status != "env_error"
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path
