import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel


class LintResult(BaseModel):
    passed: bool
    output: str


def run_linter(repo: Path, paths: list[str]) -> LintResult:
    """Lint only the given files, so pre-existing problems elsewhere in the repo don't count against a patch."""
    py_files = [p for p in paths if p.endswith(".py") and (Path(repo) / p).is_file()]
    if not py_files:
        return LintResult(passed=True, output="no python files to lint")
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--no-cache", "--output-format", "concise", *py_files],
        cwd=repo, capture_output=True, text=True, timeout=60, check=False,
    )
    return LintResult(passed=result.returncode == 0, output=(result.stdout + result.stderr).strip())
