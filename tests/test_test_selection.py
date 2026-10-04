"""Which tests count as related to a changed file, across the repo layouts SWE-bench uses."""
from pathlib import Path

from tools import filesystem, git
from tools.testing import is_test_module, select_tests


def make_repo(tmp_path: Path, files: dict[str, str]) -> tuple[Path, list[str]]:
    repo = tmp_path / "repo"
    for path, content in files.items():
        filesystem.write_file(repo, path, content)
    git._git(repo, "init", "-q")
    git._git(repo, "add", "-A")
    return repo, git.list_files(repo)


def test_only_real_test_modules_count():
    for path in ("tests/test_views.py", "pkg/views_test.py", "tests/delete/tests.py", "tests/checkers/unittest_misc.py"):
        assert is_test_module(path), path
    for path in ("tests/servers/views.py", "tests/admin_default_site/sites.py", "tests/conftest.py", "tests/__init__.py"):
        assert not is_test_module(path), path


def test_django_style_feature_dirs_win_and_the_package_name_never_matches(tmp_path):
    repo, files = make_repo(tmp_path, {
        "django/db/models/deletion.py": "class Collector: pass\n",
        "django/contrib/syndication/views.py": "class Feed: pass\n",
        "tests/delete/tests.py": "def test_delete(): pass\n",
        "tests/syndication_tests/tests.py": "def test_feed(): pass\n",
        "tests/admin_views/test_views.py": "def test_admin(): pass\n",
        "tests/generic_views/test_views.py": "def test_generic(): pass\n",
        "tests/template_backends/test_django.py": "def test_engine(): pass\n",
        "tests/servers/views.py": "def helper(): pass\n",
    })
    assert select_tests(repo, files, ["django/db/models/deletion.py"]) == ["tests/delete/tests.py"]
    assert select_tests(repo, files, ["django/contrib/syndication/views.py"]) == ["tests/syndication_tests/tests.py"]


def test_sympy_style_tests_next_to_the_code(tmp_path):
    repo, files = make_repo(tmp_path, {
        "sympy/core/power.py": "class Pow: pass\n",
        "sympy/core/tests/test_power.py": "def test_pow(): pass\n",
        "sympy/core/tests/test_arit.py": "def test_add(): pass\n",
        "sympy/sets/tests/test_power.py": "def test_powerset(): pass\n",
    })
    assert select_tests(repo, files, ["sympy/core/power.py"]) == ["sympy/core/tests/test_power.py"]


def test_flat_test_dir_found_through_imports(tmp_path):
    repo, files = make_repo(tmp_path, {
        "src/_pytest/assertion/rewrite.py": "class AssertionRewriter: pass\n",
        "testing/test_assertrewrite.py": "from _pytest.assertion.rewrite import AssertionRewriter\n",
        "testing/test_assertion.py": "import _pytest.assertion\n",
        "testing/test_collection.py": "def test_collect(): pass\n",
    })
    assert select_tests(repo, files, ["src/_pytest/assertion/rewrite.py"]) == ["testing/test_assertrewrite.py"]


def test_root_level_package_tests_as_the_fallback(tmp_path):
    repo, files = make_repo(tmp_path, {
        "requests/sessions.py": "class Session: pass\n",
        "test_requests.py": "import requests\n",
        "tests/deep/nested/test_requests.py": "def test_unrelated(): pass\n",
    })
    assert select_tests(repo, files, ["requests/sessions.py"]) == ["test_requests.py"]


def test_named_test_files_come_first_and_the_limit_holds(tmp_path):
    repo, files = make_repo(tmp_path, {
        "shop/pagination.py": "def paginate(): pass\n",
        "tests/test_pagination.py": "def test_p(): pass\n",
        "tests/test_other.py": "def test_o(): pass\n",
    })
    assert select_tests(repo, files, ["tests/test_other.py", "shop/pagination.py"]) == [
        "tests/test_other.py", "tests/test_pagination.py",
    ]
    assert select_tests(repo, files, ["tests/test_other.py", "shop/pagination.py"], limit=1) == ["tests/test_other.py"]
    assert select_tests(repo, files, ["shop/missing.py", "README.md"]) == []
