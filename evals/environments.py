"""Turning a task into a workspace, a sandbox config and test settings the agent can run against."""
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel

from agent.state import CheckSettings
from evals.tasks import Task
from sandbox.runner import SandboxConfig
from sandbox.security import SandboxLimits
from tools import git
from tools.testing import TestCommand

SWEBENCH_PLATFORM = "linux/amd64"  # published images are x86_64; Docker Desktop emulates them on Apple Silicon
SWEBENCH_ENV = {
    # Equivalent to `conda activate testbed` for running Python, set once for every command in the container.
    "PATH": "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "PYTHONDONTWRITEBYTECODE": "1",
    "HOME": "/tmp",
    "LANG": "C.UTF-8",
}

# The runner each repo uses in its SWE-bench eval script, minus the hidden test files it is given there.
_PYTEST = TestCommand(argv=["python", "-m", "pytest", "-rA", "-p", "no:cacheprovider"], run_all_when_unselected=False)
SWEBENCH_TEST_COMMANDS = {
    "django/django": TestCommand(
        argv=["python", "tests/runtests.py", "--verbosity", "2", "--settings=test_sqlite", "--parallel", "1"],
        path_style="django_label",
        run_all_when_unselected=False,
    ),
    "sympy/sympy": TestCommand(
        argv=["env", "PYTHONWARNINGS=ignore::UserWarning,ignore::SyntaxWarning", "python", "bin/test", "-C", "--verbose"],
        run_all_when_unselected=False,
    ),
}


class PreparedTask(BaseModel):
    task: Task
    sandbox: SandboxConfig
    checks: CheckSettings
    # SWE-bench: a copy of the image's /testbed. Custom: None, and the graph clones task.repo_path itself.
    workspace: str | None = None


def _docker(*args: str, timeout: float = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout, check=False)


def image_present(image: str) -> bool:
    return _docker("image", "inspect", image, timeout=60).returncode == 0


def _platform_flag(platform: str | None) -> list[str]:
    return ["--platform", platform] if platform else []


def pull_image(image: str, platform: str | None = SWEBENCH_PLATFORM) -> None:
    """Pulls only when missing. SWE-bench images are often 1-3 GB each."""
    if image_present(image):
        return
    result = _docker("pull", *_platform_flag(platform), image)
    if result.returncode != 0:
        raise RuntimeError(f"could not pull {image}: {result.stderr.strip()[-500:]}")


def copy_testbed(image: str, dest: Path, platform: str | None = SWEBENCH_PLATFORM, source: str = "/testbed") -> Path:
    """Copy the prepared repo out of the image. A clone would miss the build artifacts and installed state in it."""
    created = _docker("create", *_platform_flag(platform), image, timeout=300)
    if created.returncode != 0:
        raise RuntimeError(f"could not create a container from {image}: {created.stderr.strip()}")
    container = created.stdout.strip()
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        copied = _docker("cp", f"{container}:{source}", str(dest))
        if copied.returncode != 0:
            raise RuntimeError(f"could not copy {source} out of {image}: {copied.stderr.strip()}")
    finally:
        _docker("rm", "--force", container, timeout=120)
    return dest


def neutralize_git_config(repo: Path) -> None:
    """Replace the copied repo's .git/config with a bare one: no remotes, filters, hooks paths or fsmonitor commands."""
    (repo / ".git" / "config").write_text(
        "[core]\n\trepositoryformatversion = 0\n\tfilemode = true\n\tbare = false\n\tlogallrefupdates = true\n"
    )


def reset_to_base(repo: Path, base_commit: str) -> None:
    """Put the workspace at the task's base commit, the commit the reference patch and hidden tests are written against.

    SWE-bench images add a "SWE-bench" commit on top of the base commit (seen so far: a mode-only change marking every
    file executable, identical content), so HEAD is not the base. Untracked files, like build output, are untouched.
    """
    try:
        git._git(repo, "cat-file", "-e", f"{base_commit}^{{commit}}")
    except RuntimeError as e:
        raise RuntimeError(f"base commit {base_commit} is not in the image's repository") from e
    git._git(repo, "reset", "--quiet", "--hard", base_commit)


def freeze_baseline(repo: Path, expected_commit: str | None) -> None:
    """Make the diff show only the agent's changes.

    Untracked files already in the image (build output, egg-info) go into .git/info/exclude, and a modified tracked
    file at the start means the workspace is not the task's base commit, which would make the patch wrong.
    """
    head = git._git(repo, "rev-parse", "HEAD").strip()
    if expected_commit and head != expected_commit:
        raise RuntimeError(f"workspace is at {head}, expected base commit {expected_commit}")
    dirty = git._git(repo, "status", "--porcelain", "--untracked-files=no").strip()
    if dirty:
        raise RuntimeError(f"workspace has tracked changes before the agent started:\n{dirty[:500]}")
    untracked = git._git(repo, "ls-files", "--others", "--exclude-standard").splitlines()
    if untracked:
        exclude = repo / ".git" / "info" / "exclude"
        exclude.parent.mkdir(parents=True, exist_ok=True)
        with exclude.open("a") as f:
            f.write("\n# present in the task image before PatchPilot started\n")
            f.write("".join(f"/{path}\n" for path in untracked))


def swebench_sandbox(image: str, platform: str | None = SWEBENCH_PLATFORM) -> SandboxConfig:
    return SandboxConfig(
        image=image,
        platform=platform,
        workdir="/testbed",
        env=SWEBENCH_ENV,
        install_dependencies=False,
        # Emulated x86 is several times slower; real test suites need more room than the fixture.
        command_timeout=900,
        lifetime_seconds=4 * 3600,
        limits=SandboxLimits(cpus=2.0, memory="4g", tmp_size="1g"),
    )


def prepare(task: Task, work_dir: Path, platform: str | None = SWEBENCH_PLATFORM) -> PreparedTask:
    if task.source == "custom":
        return PreparedTask(task=task, sandbox=SandboxConfig(), checks=CheckSettings())

    pull_image(task.image, platform)
    workspace = work_dir / task.instance_id / "testbed"
    if workspace.exists():
        shutil.rmtree(workspace)
    copy_testbed(task.image, workspace, platform)
    neutralize_git_config(workspace)
    # Before indexing or any edit, so everything downstream (retrieval, diff, the patch) is relative to the base commit.
    if task.base_commit:
        reset_to_base(workspace, task.base_commit)
    freeze_baseline(workspace, task.base_commit)
    checks = CheckSettings(
        test_command=SWEBENCH_TEST_COMMANDS.get(task.repo, _PYTEST),
        # Old repos fail most of today's lint rules before any edit; the signal would be noise.
        lint=False,
    )
    return PreparedTask(task=task, sandbox=swebench_sandbox(task.image, platform), checks=checks,
                        workspace=str(workspace))
