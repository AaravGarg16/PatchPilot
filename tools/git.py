import shutil
import subprocess
from pathlib import Path

# Pinned so a user's global git config (signing, hooks identity) can't break the baseline commit.
_GIT_CONFIG = ["-c", "user.name=PatchPilot", "-c", "user.email=patchpilot@localhost", "-c", "commit.gpgsign=false"]


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *_GIT_CONFIG, *args], cwd=repo, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def checkout_repo(source: str, dest: Path) -> Path:
    """Clone a git repo, or copy a plain local directory and commit it as the baseline to diff against."""
    dest = Path(dest)
    src = Path(source)
    if src.is_dir() and not (src / ".git").exists():
        shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        _git(dest, "init", "-q")
        _git(dest, "add", "-A")
        _git(dest, "commit", "-q", "-m", "baseline")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        _git(dest.parent, "clone", "-q", source, str(dest))
    return dest


def list_files(repo: Path) -> list[str]:
    return _git(repo, "ls-files").splitlines()


def diff(repo: Path) -> str:
    # Stage everything first so newly created files show up in the diff too.
    _git(repo, "add", "-A")
    return _git(repo, "diff", "--cached")
