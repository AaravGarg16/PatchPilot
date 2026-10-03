import ast
from pathlib import PurePosixPath

from pydantic import BaseModel

LANGUAGES = {
    ".py": "python",
    ".md": "markdown",
    ".rst": "text",
    ".txt": "text",
    ".toml": "config",
    ".ini": "config",
    ".cfg": "config",
    ".yaml": "config",
    ".yml": "config",
    ".json": "config",
}
# Big enough for most functions, small enough that one chunk never dominates the code budget.
MAX_CHUNK_LINES = 120
MAX_FILE_BYTES = 200_000


class Chunk(BaseModel):
    path: str
    language: str
    symbol: str | None
    chunk_type: str  # function | method | class | module | doc_section | config | text
    start_line: int
    end_line: int
    content: str

    def label(self) -> str:
        kind = f"{self.chunk_type} {self.symbol}" if self.symbol else self.chunk_type
        return f"{self.path}:{self.start_line}-{self.end_line} ({kind})"


def language_for(path: str) -> str | None:
    return LANGUAGES.get(PurePosixPath(path).suffix.lower())


def chunk_file(path: str, text: str) -> list[Chunk]:
    language = language_for(path)
    if language is None or not text.strip():
        return []
    lines = text.splitlines()
    if language == "python":
        try:
            return _chunk_python(path, lines, ast.parse(text))
        except SyntaxError:
            return _windows(path, language, "text", None, lines, 1, len(lines))
    if language == "markdown":
        return _chunk_markdown(path, lines)
    return _windows(path, language, "config" if language == "config" else "text", None, lines, 1, len(lines))


def _windows(path, language, chunk_type, symbol, lines, start, end) -> list[Chunk]:
    """Cut lines[start..end] (1-indexed, inclusive) into chunks of at most MAX_CHUNK_LINES."""
    chunks = []
    for lo in range(start, end + 1, MAX_CHUNK_LINES):
        hi = min(lo + MAX_CHUNK_LINES - 1, end)
        content = "\n".join(lines[lo - 1:hi])
        if content.strip():
            chunks.append(Chunk(path=path, language=language, symbol=symbol, chunk_type=chunk_type,
                                start_line=lo, end_line=hi, content=content))
    return chunks


def _node_start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(d.lineno for d in decorators)])


def _chunk_python(path: str, lines: list[str], tree: ast.Module) -> list[Chunk]:
    chunks: list[Chunk] = []
    pending_module: list[ast.stmt] = []

    def flush_module():
        if pending_module:
            start, end = _node_start(pending_module[0]), pending_module[-1].end_lineno
            chunks.extend(_windows(path, "python", "module", None, lines, start, end))
            pending_module.clear()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            flush_module()
            chunks.extend(_windows(path, "python", "function", node.name, lines, _node_start(node), node.end_lineno))
        elif isinstance(node, ast.ClassDef):
            flush_module()
            chunks.extend(_chunk_class(path, lines, node))
        else:
            pending_module.append(node)
    flush_module()
    return chunks


def _chunk_class(path: str, lines: list[str], node: ast.ClassDef) -> list[Chunk]:
    # The class chunk holds the header, docstring and attributes; each method gets its own chunk, so nothing repeats.
    methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    start = _node_start(node)
    header_end = _node_start(methods[0]) - 1 if methods else node.end_lineno
    chunks = _windows(path, "python", "class", node.name, lines, start, header_end)
    for method in methods:
        chunks.extend(_windows(path, "python", "method", f"{node.name}.{method.name}", lines,
                               _node_start(method), method.end_lineno))
    return chunks


def _chunk_markdown(path: str, lines: list[str]) -> list[Chunk]:
    chunks, start, heading = [], 1, None
    in_fence = False
    for i, line in enumerate(lines, start=1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if line.startswith("#") and not in_fence and i > start:
            chunks.extend(_windows(path, "markdown", "doc_section", heading, lines, start, i - 1))
            start = i
        if line.startswith("#") and not in_fence and i == start:
            heading = line.lstrip("#").strip() or None
    chunks.extend(_windows(path, "markdown", "doc_section", heading, lines, start, len(lines)))
    return chunks
