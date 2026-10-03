import asyncio

import pytest

from tests.fakes import FIXED_PAGINATION, FIXTURE_REPO
from tools import filesystem, git, linting, testing
from tools.mcp_client import repo_tools


@pytest.fixture
def repo(tmp_path):
    return git.checkout_repo(FIXTURE_REPO, tmp_path / "repo")


def test_checkout_of_plain_dir_creates_clean_baseline(repo):
    assert "shop/pagination.py" in git.list_files(repo)
    assert git.diff(repo) == ""


def test_checkout_clones_git_repos(repo, tmp_path):
    clone = git.checkout_repo(str(repo), tmp_path / "clone")
    assert git.list_files(clone) == git.list_files(repo)


def test_diff_includes_edits_and_new_files(repo):
    filesystem.write_file(repo, "shop/pagination.py", FIXED_PAGINATION)
    filesystem.write_file(repo, "shop/new_module.py", "X = 1\n")
    diff = git.diff(repo)
    assert "+    start = (page - 1) * per_page" in diff
    assert "shop/new_module.py" in diff


@pytest.mark.parametrize("bad_path", ["../outside.py", "/etc/passwd", ".git/config", "."])
def test_paths_outside_repo_are_refused(repo, bad_path):
    with pytest.raises(ValueError):
        filesystem.resolve_in_repo(repo, bad_path)


def test_run_tests_reports_failure_then_pass(repo):
    before = testing.run_tests(repo)
    assert not before.passed
    assert "3 failed" in before.output

    filesystem.write_file(repo, "shop/pagination.py", FIXED_PAGINATION)
    after = testing.run_tests(repo)
    assert after.passed, after.output


def test_run_tests_does_not_pass_secrets_to_repo_code(repo, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    filesystem.write_file(repo, "shop/pagination.py", FIXED_PAGINATION)
    filesystem.write_file(
        repo, "tests/test_env.py", "import os\n\ndef test_no_key():\n    assert 'ANTHROPIC_API_KEY' not in os.environ\n"
    )
    result = testing.run_tests(repo)
    assert result.passed, result.output


def test_run_tests_times_out(repo):
    filesystem.write_file(repo, "tests/test_slow.py", "import time\n\ndef test_slow():\n    time.sleep(30)\n")
    result = testing.run_tests(repo, timeout=2)
    assert result.timed_out
    assert not result.passed


def test_edit_file_replaces_one_exact_match(repo):
    filesystem.edit_file(repo, "shop/pagination.py", "start = page * per_page", "start = (page - 1) * per_page")
    assert "start = (page - 1) * per_page" in filesystem.read_file(repo, "shop/pagination.py")


@pytest.mark.parametrize("old", ["not in the file", "per_page", ""])
def test_edit_file_refuses_missing_or_ambiguous_matches(repo, old):
    with pytest.raises(ValueError, match="exactly once"):
        filesystem.edit_file(repo, "shop/pagination.py", old, "x")


def test_edit_file_refuses_existing_tests(repo):
    with pytest.raises(ValueError, match="existing test files"):
        filesystem.edit_file(repo, "tests/test_pagination.py", "import pytest", "")


def test_create_file_refuses_to_overwrite(repo):
    filesystem.create_file(repo, "tests/test_new.py", "def test_x():\n    pass\n")
    with pytest.raises(ValueError, match="already exists"):
        filesystem.create_file(repo, "shop/pagination.py", "")


def test_list_directory_hides_git(repo):
    assert filesystem.list_directory(repo) == [".gitignore", "README.md", "pytest.ini", "shop/", "tests/"]
    assert "shop/pagination.py" in filesystem.list_directory(repo, "shop")


def test_search_code_finds_matches_with_line_numbers(repo):
    assert git.search_code(repo, r"def (paginate|total_pages)").splitlines() == [
        "shop/pagination.py:1:def total_pages(total_items: int, per_page: int) -> int:",
        "shop/pagination.py:7:def paginate(items: list, page: int, per_page: int) -> list:",
    ]
    assert git.search_code(repo, "no_such_symbol") == "no matches"


def test_linter_only_checks_given_files(repo):
    filesystem.write_file(repo, "shop/messy.py", "import os\n")
    assert linting.run_linter(repo, ["shop/pagination.py"]).passed
    result = linting.run_linter(repo, ["shop/messy.py"])
    assert not result.passed and "F401" in result.output


def test_mcp_server_advertises_and_runs_tools(repo):
    async def exercise():
        async with repo_tools(repo) as tools:
            by_name = {tool.name: tool for tool in tools}
            read = await by_name["read_file"].ainvoke({"path": "shop/formatting.py"})
            blocked = await by_name["read_file"].ainvoke(
                {"type": "tool_call", "id": "1", "name": "read_file", "args": {"path": "../../etc/passwd"}}
            )
            return sorted(by_name), str(read), blocked

    names, read, blocked = asyncio.run(exercise())
    assert names == ["create_file", "edit_file", "git_diff", "list_directory", "read_file", "run_linter", "run_tests", "search_code"]
    assert "def format_price" in read
    assert blocked.status == "error" and "outside the repository" in str(blocked.content)
