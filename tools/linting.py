from pathlib import Path

from pydantic import BaseModel

from sandbox.runner import Sandbox


class LintResult(BaseModel):
    passed: bool
    output: str


def run_linter(sandbox: Sandbox, repo: Path, paths: list[str]) -> LintResult:
    """Lint only the given files, so pre-existing problems elsewhere in the repo don't count against a patch.

    Ruff runs in the sandbox too: it reads the repo's own config, and nothing repo-controlled runs on the host.
    """
    py_files = [p for p in paths if p.endswith(".py") and (Path(repo) / p).is_file()]
    if not py_files:
        return LintResult(passed=True, output="no python files to lint")
    result = sandbox.exec(["ruff", "check", "--no-cache", "--output-format", "concise", *py_files], timeout=60)
    return LintResult(passed=result.exit_code == 0, output=result.output.strip())
