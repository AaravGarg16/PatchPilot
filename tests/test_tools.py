import pytest

from tests.fakes import FIXED_PAGINATION, FIXTURE_REPO
from tools import filesystem, git, testing


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
