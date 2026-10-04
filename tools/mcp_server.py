"""MCP server exposing the executor's tools for one repository.

Run as `python -m tools.mcp_server <repo_path> <sandbox_name> <checks_json>`; the agent connects over stdio and
discovers the tools. File tools work on the host workspace; tests and lint execute in the run's sandbox container.
"""
import json
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from sandbox.runner import Sandbox
from tools import filesystem, git, linting, testing

MAX_OUTPUT_CHARS = 6000


def _tail(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return "[earlier output truncated]\n" + text[-MAX_OUTPUT_CHARS:]


def build_server(repo: Path, sandbox: Sandbox, checks: dict | None = None) -> FastMCP:
    checks = checks or {}
    command = testing.TestCommand(**checks.get("test_command", {}))
    lint_enabled = checks.get("lint", True)
    server = FastMCP("patchpilot-repo", log_level="WARNING")

    @server.tool()
    def read_file(path: str, start_line: int = 1, max_lines: int | None = None) -> str:
        """Read a file in the repository. Paths are relative to the repo root. For large files, pass start_line and max_lines to read a slice."""
        return filesystem.read_file(repo, path, start_line, max_lines)

    @server.tool()
    def list_directory(path: str = ".") -> str:
        """List files and subdirectories (suffixed with /) of a directory in the repository."""
        return "\n".join(filesystem.list_directory(repo, path)) or "(empty)"

    @server.tool()
    def search_code(pattern: str) -> str:
        """Search the repository with an extended regex. Returns path:line:text matches."""
        return git.search_code(repo, pattern)

    @server.tool()
    def edit_file(path: str, old_string: str, new_string: str) -> str:
        """Replace an exact snippet in an existing file. old_string must appear exactly once; include enough surrounding lines to make it unique. Existing test files cannot be edited."""
        filesystem.edit_file(repo, path, old_string, new_string)
        return f"edited {path}"

    @server.tool()
    def create_file(path: str, content: str) -> str:
        """Create a new file with the given content. Fails if the file already exists."""
        filesystem.create_file(repo, path, content)
        return f"created {path}"

    @server.tool()
    def run_tests(paths: list[str] | None = None) -> str:
        """Run tests in an isolated sandbox and return pass/fail plus the output. Pass test file paths to run just those; otherwise tests related to the files changed so far are run (or the whole suite, for small repos)."""
        if not paths and not command.run_all_when_unselected:
            paths = testing.select_tests(repo, git.list_files(repo), git.changed_files(repo))
        result = testing.run_tests(sandbox, command, paths)
        if not result.ran:
            return f"NOT RUN\n{result.output}. Pass test file paths explicitly."
        return f"{'PASSED' if result.passed else 'FAILED'} ({', '.join(result.selected) or 'full suite'})\n{_tail(result.output)}"

    @server.tool()
    def run_linter() -> str:
        """Lint the files changed so far."""
        if not lint_enabled:
            return "CLEAN\nlinting is disabled for this repository"
        result = linting.run_linter(sandbox, repo, git.changed_files(repo))
        return f"{'CLEAN' if result.passed else 'PROBLEMS'}\n{_tail(result.output)}"

    @server.tool()
    def git_diff() -> str:
        """Show the diff of all changes made so far."""
        return git.diff(repo) or "(no changes)"

    return server


if __name__ == "__main__":
    checks = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    build_server(Path(sys.argv[1]).resolve(), Sandbox.attach(sys.argv[2]), checks).run(transport="stdio")
