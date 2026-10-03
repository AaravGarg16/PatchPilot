from retrieval.chunker import MAX_CHUNK_LINES, chunk_file

SOURCE = '''"""Module doc."""
import math

RATE = 3


@cache
def top(a):
    return a


class Cart:
    """A cart."""

    items: list

    def add(self, x):
        self.items.append(x)

    async def total(self):
        return sum(self.items)


if __name__ == "__main__":
    top(1)
'''


def labels(chunks):
    return [(c.chunk_type, c.symbol, c.start_line, c.end_line) for c in chunks]


def test_python_chunks_follow_symbols():
    assert labels(chunk_file("shop/cart.py", SOURCE)) == [
        ("module", None, 1, 4),
        ("function", "top", 7, 9),
        ("class", "Cart", 12, 16),
        ("method", "Cart.add", 17, 18),
        ("method", "Cart.total", 20, 21),
        ("module", None, 24, 25),
    ]


def test_decorators_stay_with_their_function():
    (function,) = [c for c in chunk_file("shop/cart.py", SOURCE) if c.symbol == "top"]
    assert function.content.startswith("@cache\ndef top")


def test_long_functions_are_split_but_keep_their_symbol():
    body = "\n".join(f"    x{i} = {i}" for i in range(MAX_CHUNK_LINES + 30))
    chunks = chunk_file("big.py", f"def big():\n{body}\n")
    assert [(c.symbol, c.start_line) for c in chunks] == [("big", 1), ("big", MAX_CHUNK_LINES + 1)]


def test_unparseable_python_falls_back_to_line_windows():
    chunks = chunk_file("broken.py", "def broken(:\n    pass\n")
    assert labels(chunks) == [("text", None, 1, 2)]


def test_markdown_splits_on_headings_but_not_inside_code_fences():
    md = "# Title\nintro\n\n## Install\n```\n# not a heading\n```\n## Usage\nrun it\n"
    assert labels(chunk_file("README.md", md)) == [
        ("doc_section", "Title", 1, 3),
        ("doc_section", "Install", 4, 7),
        ("doc_section", "Usage", 8, 9),
    ]


def test_config_files_are_one_block_and_unknown_files_are_skipped():
    assert labels(chunk_file("pytest.ini", "[pytest]\ntestpaths = tests\n")) == [("config", None, 1, 2)]
    assert chunk_file("logo.png", "binary-ish") == []
    assert chunk_file("empty.py", "\n\n") == []
