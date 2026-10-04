import os
import re
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel

from sandbox.runner import Sandbox
from tools import git
from tools.filesystem import is_test_file

# Selected test files per run. Each one can take a while under emulation, so keep the set small.
MAX_SELECTED_TEST_FILES = 6
# A shared word in a file name alone (score 3) is too weak: every test_views.py would match any views.py.
MIN_RELATED_SCORE = 4
# A parent package matching this few test directories has a test dir of its own (django's syndication_tests).
FEATURE_DIR_MAX_MATCHES = 2


class TestCommand(BaseModel):
    """How a repo runs its tests. Paths are appended, converted to the runner's own style first."""

    __test__ = False

    argv: list[str] = ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    # "file": pass test file paths as-is. "django_label": tests/a/test_b.py becomes a.test_b for runtests.py.
    path_style: Literal["file", "django_label"] = "file"
    # Small repos run the whole suite when no files are given. Large ones only run selected files.
    run_all_when_unselected: bool = True


class TestResult(BaseModel):
    __test__ = False  # stop pytest from trying to collect this as a test class

    passed: bool
    exit_code: int | None
    output: str
    timed_out: bool = False
    # False when nothing was selected to run; then `passed` says nothing about the patch.
    ran: bool = True
    selected: list[str] = []


def _as_argument(path: str, style: str) -> str:
    if style == "django_label":
        parts = list(PurePosixPath(path).with_suffix("").parts)
        return ".".join(parts[1:] if parts and parts[0] == "tests" else parts)
    return path


_TEST_MODULE = re.compile(r"^(test_\w+|\w+_test|tests|unittest_\w+)\.py$")
# Directory names that say "this is a test tree" without saying what is tested.
_GENERIC_DIRS = {"tests", "test", "testing", "unit", "unittests", "integration", "src", "lib"}


def is_test_module(path: str) -> bool:
    """A module a test runner collects, not a helper that happens to live under tests/ (views.py, conftest.py)."""
    return bool(_TEST_MODULE.match(PurePosixPath(path).name))


def _tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[_\-.]+", name.lower()) if t and t not in ("test", "tests", "unittest")}


def _affine(a: str, b: str) -> bool:
    """Same word, or the same stem (delete/deletion, site/sites), or one long name inside the other."""
    if a == b:
        return True
    prefix = len(os.path.commonprefix([a, b]))
    if min(len(a), len(b)) >= 4 and prefix >= max(4, 0.7 * min(len(a), len(b))):
        return True
    return (len(a) >= 5 and a in b) or (len(b) >= 5 and b in a)


def _any_affine(words: set[str], others: set[str]) -> bool:
    return any(_affine(w, o) for w in words for o in others)


def _dotted_module(path: str) -> str | None:
    parts = list(PurePosixPath(path).with_suffix("").parts)
    while parts and parts[0] in ("src", "lib"):
        parts.pop(0)
    if parts and parts[-1] == "__init__":
        return None  # a package: everything imports it, so imports say nothing about which tests matter
    return ".".join(parts) if len(parts) > 1 else None


def _importers(repo: Path, module: str) -> set[str]:
    # POSIX classes, not \s, \w or \b: git grep -E uses the platform's ERE, which may not know them.
    def esc(name: str) -> str:
        return name.replace(".", "\\.")

    parent, leaf = module.rsplit(".", 1)
    end = "([^.[:alnum:]_]|$)"
    pattern = (
        f"(from|import)[[:space:]]+{esc(module)}{end}"
        f"|from[[:space:]]+{esc(parent)}[[:space:]]+import[[:space:]](.*[^[:alnum:]_])?{esc(leaf)}([^[:alnum:]_]|$)"
    )
    return set(git.grep_files(repo, pattern))


def _score_tests(repo: Path, source: str, test_modules: list[str], top_package: str) -> dict[str, int]:
    """Score how related each test module is to one changed source file. See select_tests for the signals."""
    path = PurePosixPath(source)
    stem = path.stem if path.stem != "__init__" else path.parent.name
    stem_words = _tokens(stem)
    parent_words = _tokens(path.parent.name) - {top_package} - _GENERIC_DIRS
    source_dir_words = set().union(set(), *(_tokens(d) for d in path.parts[:-1])) - {top_package} - _GENERIC_DIRS
    module = _dotted_module(source)
    importers = _importers(repo, module) if module else set()

    def dir_words_of(test: str) -> set[str]:
        parts = PurePosixPath(test).parts[:-1]
        return set().union(set(), *(_tokens(d) for d in parts if d.lower() not in _GENERIC_DIRS)) - {top_package}

    test_dirs = {PurePosixPath(t).parent: dir_words_of(t) for t in test_modules}
    parent_dir_matches = sum(1 for words in test_dirs.values() if parent_words and _any_affine(parent_words, words))
    parent_weight = 6 if parent_dir_matches <= FEATURE_DIR_MAX_MATCHES else 4

    scores: dict[str, int] = {}
    for test in test_modules:
        t = PurePosixPath(test)
        dir_words = test_dirs[t.parent]
        file_words = _tokens(t.stem)
        score = 0
        # An exact name only means "tests for this module" in a generic test dir or one related to the source's dir;
        # in a flat tree like Django's, tests/admin_docs/test_views.py says nothing about contrib/syndication/views.py.
        same_area = not dir_words or _any_affine(dir_words, source_dir_words)
        if same_area and t.stem in (f"test_{stem}", f"{stem}_test", f"tests_{stem}", f"unittest_{stem}"):
            score += 10
        if _any_affine(stem_words, dir_words):
            score += 5
        if test in importers:
            score += 4
        if parent_words and _any_affine(parent_words, dir_words):
            score += parent_weight
        if t.stem == f"test_{top_package}" and len(t.parts) <= 2:
            score += 4
        if same_area and _any_affine(stem_words, file_words):
            score += 3
        if score:
            scores[test] = score
    return scores


def select_tests(repo: Path, repo_files: list[str], hints: list[str], limit: int = MAX_SELECTED_TEST_FILES) -> list[str]:
    """Test modules related to the given files: test modules named directly, then the best matches for each source.

    Per source file only the top-scoring tests are kept, so a weak signal (a shared word in a file name) never fills
    the list when a strong one (an exact test_<module>.py, a test directory named for the module, an import of it)
    exists. The repo's top-level package name never counts as a match, so "django" in a path relates nothing.
    """
    test_modules = [f for f in repo_files if f.endswith(".py") and is_test_module(f)]
    selected: list[str] = [h for h in hints if h in test_modules]
    for hint in hints:
        if not hint.endswith(".py") or is_test_file(hint) or hint not in repo_files:
            continue
        package_parts = [p for p in PurePosixPath(hint).parts[:-1] if p not in ("src", "lib")]
        scores = _score_tests(repo, hint, test_modules, package_parts[0] if package_parts else "")
        best = max(scores.values(), default=0)
        if best < MIN_RELATED_SCORE:
            continue
        selected.extend(sorted((t for t, s in scores.items() if s == best), key=lambda t: (len(t), t)))
    return list(dict.fromkeys(selected))[:limit]


def run_tests(
    sandbox: Sandbox, command: TestCommand | None = None, paths: list[str] | None = None, timeout: int | None = None
) -> TestResult:
    """Run tests inside the sandbox container: the given files, or the whole suite if the command allows it."""
    command = command or TestCommand()
    paths = paths or []
    if not paths and not command.run_all_when_unselected:
        return TestResult(passed=False, exit_code=None, ran=False,
                          output="no related test files were found, so no tests were run")
    argv = [*command.argv, *dict.fromkeys(_as_argument(p, command.path_style) for p in paths)]
    result = sandbox.exec(argv, timeout)
    # pytest exits 5 when it collects nothing; that is not a pass.
    return TestResult(passed=result.exit_code == 0 and not result.timed_out, exit_code=result.exit_code,
                      output=result.output, timed_out=result.timed_out, selected=paths)
