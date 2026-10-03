"""Context engineering: a token budget per prompt section, and the packing and compression that enforce it."""
import math
from pathlib import Path

from langchain_core.messages import BaseMessage, ToolMessage
from pydantic import BaseModel

from retrieval.retriever import ScoredChunk

DEFAULT_BUDGET_TOKENS = 40_000

# Shares from the spec. Each prompt only uses the sections it needs, so real prompts sit under the total.
SHARES = {
    "system": 0.10,
    "task": 0.10,
    "plan": 0.10,
    "code": 0.45,
    "failures": 0.15,
    "tool_outputs": 0.10,
}


class ContextBudget(BaseModel):
    total_tokens: int = DEFAULT_BUDGET_TOKENS

    def tokens(self, section: str) -> int:
        return int(self.total_tokens * SHARES[section])


def estimate_tokens(text: str) -> int:
    # ~4 chars per token for English and code. A heuristic, but it needs no API call and errs on the high side for code.
    return math.ceil(len(text) / 4)


def fit(text: str, max_tokens: int, keep: str = "head") -> str:
    """Trim text to the budget, keeping the start (task, plan) or the end (test output, where failures summarize)."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    if keep == "tail":
        return f"[{len(text) - max_chars} earlier chars trimmed]\n" + text[-max_chars:]
    return text[:max_chars] + f"\n[{len(text) - max_chars} more chars trimmed]"


def render_file(path: str, content: str) -> str:
    return f"### {path} (whole file)\n```\n{content}\n```"


def render_chunk(chunk: ScoredChunk) -> str:
    return f"### {chunk.label()}\n```\n{chunk.content}\n```"


def build_code_context(repo: Path, files: list[str], chunks: list[ScoredChunk], max_tokens: int) -> str:
    """Selected-file retrieval: whole files when they fit, otherwise their best chunks, then other chunks by score.

    Whole files are read from the workspace at call time, so they include edits made since indexing.
    """
    blocks: list[str] = []
    remaining = max_tokens
    shown: set[int] = set()

    def add(block: str) -> bool:
        nonlocal remaining
        cost = estimate_tokens(block)
        if cost > remaining:
            return False
        blocks.append(block)
        remaining -= cost
        return True

    for path in dict.fromkeys(files):
        file = Path(repo) / path
        if file.is_file() and add(render_file(path, file.read_text())):
            shown.update(c.id for c in chunks if c.path == path)
            continue
        for chunk in chunks:
            if chunk.path == path and chunk.id not in shown and add(render_chunk(chunk)):
                shown.add(chunk.id)

    for chunk in chunks:
        if chunk.id not in shown and add(render_chunk(chunk)):
            shown.add(chunk.id)
    return "\n\n".join(blocks) or "(no code retrieved)"


def compress_tool_outputs(messages: list[BaseMessage], max_tokens: int) -> int:
    """Elide older tool results once they exceed the budget, newest first. The latest result is always kept.

    Returns how many results were elided. The tool call itself stays, so the model knows it can call again.
    """
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    used = elided = 0
    for i, message in enumerate(reversed(tool_messages)):
        content = message.content if isinstance(message.content, str) else str(message.content)
        if content.startswith("[elided"):
            continue
        cost = estimate_tokens(content)
        if i == 0 or used + cost <= max_tokens:
            used += cost
            continue
        message.content = f"[elided {message.name or 'tool'} output (~{cost} tokens) to save context; call again if needed]"
        elided += 1
    return elided



def _status(ok: bool) -> str:
    return "passed" if ok else "failed"


def _problems(record) -> str:
    return "\n".join(f"- {p}" for p in record.problems) or "- (evaluator listed none)"


def render_failures(history: list, max_tokens: int) -> str:
    """History summarization: the latest failure in full, older ones reduced to their problems, oldest dropped first."""
    if not history:
        return ""
    *older, latest = history
    lint = "" if latest.lint_passed else f"Lint:\n```\n{fit(latest.lint_output, max_tokens // 8, keep='tail')}\n```\n"
    full = (
        f"### Attempt {latest.attempt}\n"
        f"Tests {_status(latest.tests_passed)}:\n```\n{fit(latest.test_output, max_tokens // 2, keep='tail')}\n```\n"
        f"{lint}Evaluator problems:\n{_problems(latest)}"
    )
    compact = [f"### Attempt {r.attempt} (summary)\nTests {_status(r.tests_passed)}. Problems:\n{_problems(r)}" for r in older]
    while compact and estimate_tokens("\n\n".join([*compact, full])) > max_tokens:
        compact.pop(0)
    return fit("\n\n".join([*compact, full]), max_tokens)
