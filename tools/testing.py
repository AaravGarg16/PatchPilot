from pydantic import BaseModel

from sandbox.runner import Sandbox


class TestResult(BaseModel):
    __test__ = False  # stop pytest from trying to collect this as a test class

    passed: bool
    exit_code: int | None
    output: str
    timed_out: bool = False


def run_tests(sandbox: Sandbox, timeout: int | None = None) -> TestResult:
    """Run the repo's pytest suite inside the sandbox container."""
    result = sandbox.exec(["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"], timeout)
    # pytest exits 5 when it collects nothing; that is not a pass.
    return TestResult(passed=result.exit_code == 0 and not result.timed_out, exit_code=result.exit_code,
                      output=result.output, timed_out=result.timed_out)
