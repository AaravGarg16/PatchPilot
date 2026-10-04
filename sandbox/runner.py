import shlex
import subprocess
from pathlib import Path

from pydantic import BaseModel

from sandbox.security import SandboxLimits, hardening_flags

DEFAULT_IMAGE = "patchpilot-sandbox:latest"
BUILD_HINT = "docker build -t patchpilot-sandbox:latest sandbox/"
LABEL = "patchpilot.run"
REQUIREMENT_FILES = ["requirements.txt", "requirements-dev.txt", "requirements-test.txt"]
# Bounds how much output a misbehaving test run can push back to the host.
MAX_OUTPUT_BYTES = 200_000
# Extra host-side wait on top of the in-container timeout before we give up on docker itself.
HOST_GRACE_SECONDS = 15


class SandboxConfig(BaseModel):
    image: str = DEFAULT_IMAGE
    # e.g. "linux/amd64" for SWE-bench images, which run under emulation on Apple Silicon.
    platform: str | None = None
    # Where the workspace is mounted. SWE-bench images expect the repo at /testbed (their editable installs point there).
    workdir: str = "/workspace"
    env: dict[str, str] = {}
    # SWE-bench images ship with the environment already installed, so they skip the network phase entirely.
    install_dependencies: bool = True
    limits: SandboxLimits = SandboxLimits()
    command_timeout: int = 120
    install_timeout: int = 600
    # The container's own process exits after this, and --rm deletes it, even if PatchPilot crashes mid-run.
    lifetime_seconds: int = 3600


class ExecResult(BaseModel):
    exit_code: int | None
    output: str
    timed_out: bool = False


def _docker(*args: str, timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout, check=False)


def container_name(run_id: str) -> str:
    return f"patchpilot-{run_id}"


class Sandbox:
    """One disposable, locked-down container per agent run. The workspace is mounted read-only."""

    def __init__(self, name: str, config: SandboxConfig | None = None):
        self.name = name
        self.config = config or SandboxConfig()

    @classmethod
    def start(cls, repo: Path, run_id: str, config: SandboxConfig | None = None) -> "Sandbox":
        config = config or SandboxConfig()
        if _docker("image", "inspect", config.image).returncode != 0:
            raise RuntimeError(f"sandbox image {config.image} not found; build it with: {BUILD_HINT}")
        needs_network = config.install_dependencies and any((Path(repo) / f).is_file() for f in REQUIREMENT_FILES)
        name = container_name(run_id)
        platform = ["--platform", config.platform] if config.platform else []
        env = [arg for key, value in config.env.items() for arg in ("--env", f"{key}={value}")]
        result = _docker(
            "run", "--detach", "--rm",
            "--name", name,
            "--label", f"{LABEL}={run_id}",
            *platform,
            # Network only while dependencies install; cut right after. Repos with nothing to install never get it.
            "--network", "bridge" if needs_network else "none",
            *hardening_flags(config.limits),
            *env,
            "--workdir", config.workdir,
            "--volume", f"{Path(repo).resolve()}:{config.workdir}:ro",
            config.image,
            "sleep", str(config.lifetime_seconds),
        )
        if result.returncode != 0:
            raise RuntimeError(f"could not start sandbox: {result.stderr.strip()}")
        sandbox = cls(name, config)
        if needs_network:
            try:
                sandbox.install_dependencies(repo)
            finally:
                sandbox.disable_network()
        return sandbox

    @classmethod
    def attach(cls, name: str, config: SandboxConfig | None = None) -> "Sandbox":
        return cls(name, config)

    def exec(self, cmd: list[str], timeout: int | None = None) -> ExecResult:
        timeout = timeout or self.config.command_timeout
        # timeout runs inside the container so the process really dies; tail caps output; pipefail keeps the exit code.
        # --verbose prints a marker, which is how a timeout is told apart from an out-of-memory kill (both can exit 137).
        script = f"timeout --verbose -k 5 {int(timeout)} {shlex.join(cmd)} 2>&1 | tail -c {MAX_OUTPUT_BYTES}"
        try:
            result = _docker("exec", self.name, "bash", "-o", "pipefail", "-c", script,
                             timeout=timeout + HOST_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            _docker("kill", self.name)
            return ExecResult(exit_code=None, output=f"timed out after {timeout}s; sandbox killed", timed_out=True)
        output = result.stdout + result.stderr
        timed_out = "timeout: sending signal" in output
        if timed_out:
            output += f"\ntimed out after {timeout}s"
        return ExecResult(exit_code=result.returncode, output=output, timed_out=timed_out)

    def install_dependencies(self, repo: Path) -> ExecResult:
        files = [f for f in REQUIREMENT_FILES if (Path(repo) / f).is_file()]
        args = [arg for f in files for arg in ("-r", f)]
        return self.exec(
            ["pip", "install", "--no-cache-dir", "--disable-pip-version-check", "--target", "/deps", *args],
            timeout=self.config.install_timeout,
        )

    def disable_network(self) -> None:
        _docker("network", "disconnect", "--force", "bridge", self.name)

    def is_running(self) -> bool:
        result = _docker("inspect", "--format", "{{.State.Running}}", self.name)
        return result.returncode == 0 and result.stdout.strip() == "true"

    def destroy(self) -> None:
        _docker("rm", "--force", self.name)


def destroy_run(run_id: str) -> None:
    """Safety net: remove any container left over from a run, whatever state it is in."""
    ids = _docker("ps", "--all", "--quiet", "--filter", f"label={LABEL}={run_id}").stdout.split()
    if ids:
        _docker("rm", "--force", *ids)
