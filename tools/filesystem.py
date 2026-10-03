from pathlib import Path


def resolve_in_repo(repo: Path, rel_path: str) -> Path:
    """Resolve a model-supplied path, refusing anything outside the repo or inside .git."""
    root = Path(repo).resolve()
    target = (root / rel_path).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError(f"path is outside the repository: {rel_path}")
    if ".git" in target.relative_to(root).parts:
        raise ValueError(f"path is inside .git: {rel_path}")
    return target


def read_file(repo: Path, rel_path: str) -> str:
    return resolve_in_repo(repo, rel_path).read_text()


def write_file(repo: Path, rel_path: str, content: str) -> None:
    target = resolve_in_repo(repo, rel_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
