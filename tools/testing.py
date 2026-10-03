import os
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel

DEFAULT_TIMEOUT_SECONDS = 120


class TestResult(BaseModel):
    __test__ = False  # stop pytest from trying to collect this as a test class

    passed: bool
    exit_code: int | None
    output: str
    timed_out: bool = False


def run_tests(repo: Path, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> TestResult:
    """Run the repo's pytest suite in a subprocess. Not a sandbox: only for trusted repos until Phase 5."""
    # Minimal env so API keys and other secrets from .env never reach repository code.
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    try:
        result = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        output = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        return TestResult(passed=False, exit_code=None, output=output + f"\nTimed out after {timeout}s", timed_out=True)
    # pytest exits 5 when it collects nothing; that is not a pass.
    return TestResult(passed=result.returncode == 0, exit_code=result.returncode, output=result.stdout + result.stderr)
