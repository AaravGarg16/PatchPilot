from pathlib import Path, PurePosixPath

# Bounds what a single tool call can push into the model's context.
MAX_READ_CHARS = 100_000


def resolve_in_repo(repo: Path, rel_path: str) -> Path:
    """Resolve a model-supplied path, refusing anything outside the repo or inside .git."""
    root = Path(repo).resolve()
    target = (root / rel_path).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError(f"path is outside the repository: {rel_path}")
    if ".git" in target.relative_to(root).parts:
        raise ValueError(f"path is inside .git: {rel_path}")
    return target


def is_test_file(rel_path: str) -> bool:
    p = PurePosixPath(rel_path)
    return (
        "tests" in p.parts
        or "test" in p.parts
        or p.name.startswith("test_")
        or p.name.endswith("_test.py")
        or p.name == "conftest.py"
    )


def _guard_existing_test(repo: Path, target: Path) -> None:
    # Rewriting an existing test is the easiest way to "pass", so the tools never allow it.
    rel = target.relative_to(Path(repo).resolve()).as_posix()
    if target.exists() and is_test_file(rel):
        raise ValueError(f"{rel}: existing test files cannot be modified")


def read_file(repo: Path, rel_path: str) -> str:
    target = resolve_in_repo(repo, rel_path)
    if not target.is_file():
        raise ValueError(f"not a file: {rel_path}")
    content = target.read_text()
    if len(content) > MAX_READ_CHARS:
        return content[:MAX_READ_CHARS] + f"\n[truncated: file is {len(content)} chars]"
    return content


def write_file(repo: Path, rel_path: str, content: str) -> None:
    target = resolve_in_repo(repo, rel_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


def create_file(repo: Path, rel_path: str, content: str) -> None:
    target = resolve_in_repo(repo, rel_path)
    if target.exists():
        raise ValueError(f"{rel_path} already exists; use edit_file")
    write_file(repo, rel_path, content)


def edit_file(repo: Path, rel_path: str, old_string: str, new_string: str) -> None:
    """Replace one exact, unique occurrence of old_string."""
    target = resolve_in_repo(repo, rel_path)
    if not target.is_file():
        raise ValueError(f"not a file: {rel_path}")
    _guard_existing_test(repo, target)
    content = target.read_text()
    count = content.count(old_string) if old_string else 0
    if count != 1:
        raise ValueError(f"old_string must match exactly once in {rel_path}, found {count} matches")
    target.write_text(content.replace(old_string, new_string, 1))


def list_directory(repo: Path, rel_path: str = ".") -> list[str]:
    root = Path(repo).resolve()
    target = root if rel_path in ("", ".") else resolve_in_repo(repo, rel_path)
    if not target.is_dir():
        raise ValueError(f"not a directory: {rel_path}")
    entries = []
    for child in sorted(target.iterdir()):
        if child.name in (".git", "__pycache__"):
            continue
        rel = child.relative_to(root).as_posix()
        entries.append(rel + "/" if child.is_dir() else rel)
    return entries
