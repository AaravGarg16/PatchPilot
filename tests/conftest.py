import asyncio
import os
import subprocess
import time

import psycopg
import pytest

from agent.run import run_issue
from retrieval.retriever import Retriever
from retrieval.store import ChunkStore
from sandbox.runner import BUILD_HINT, DEFAULT_IMAGE, Sandbox, image_exists
from tests.fakes import FAKE_MODEL, FIXTURE_REPO, HashEmbedder
from tools import git

# A separate database so tests never touch the dev index. Created once with: createdb patchpilot_test
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "postgresql://localhost:5432/patchpilot_test")


@pytest.fixture(autouse=True)
def langfuse_off(monkeypatch):
    """Tests never send traces to a real Langfuse project, whatever keys are in .env.

    An explicit off switch rather than deleting keys: load_dotenv() never overrides a variable that is already set,
    so nothing later in a test can quietly turn tracing back on. Tests that exercise tracing opt in with fakes.
    """
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")


@pytest.fixture(autouse=True)
def anthropic_unreachable(monkeypatch):
    """No test may reach Anthropic's API. If one tries, the call fails against a dead local port instead."""
    for variable in ("ANTHROPIC_API_URL", "ANTHROPIC_BASE_URL"):
        monkeypatch.setenv(variable, "http://127.0.0.1:9")


@pytest.fixture(scope="session")
def test_db_url() -> str:
    try:
        psycopg.connect(TEST_DATABASE_URL, connect_timeout=3).close()
    except psycopg.OperationalError as e:
        pytest.fail(
            f"Postgres test database not reachable at {TEST_DATABASE_URL}: {e}\n"
            "Start it with `brew services start postgresql@17` and create it with `createdb patchpilot_test`."
        )
    return TEST_DATABASE_URL


@pytest.fixture
def retriever(test_db_url) -> Retriever:
    embedder = HashEmbedder()
    return Retriever(ChunkStore(test_db_url, embedder.dim), embedder)


def _docker_ok(*args: str, attempts: int = 5) -> bool:
    # Docker Desktop can fail its first calls after sitting idle; a failed probe would take down every Docker test.
    for attempt in range(attempts):
        if subprocess.run(["docker", *args], capture_output=True, check=False).returncode == 0:
            return True
        time.sleep(2 * (attempt + 1))
    return False


@pytest.fixture(scope="session")
def sandbox_image() -> str:
    """Repository code only ever runs in the sandbox, so tests that run it need Docker and the image."""
    if not _docker_ok("info"):
        pytest.fail("Docker is not running. Start Docker Desktop (open -a Docker) and rerun.")
    if not image_exists(DEFAULT_IMAGE):
        pytest.fail(f"Sandbox image {DEFAULT_IMAGE} is missing. Build it once with: {BUILD_HINT}")
    return DEFAULT_IMAGE


@pytest.fixture
def repo(tmp_path):
    return git.checkout_repo(FIXTURE_REPO, tmp_path / "repo")


@pytest.fixture
def sandbox(sandbox_image, repo, tmp_path):
    box = Sandbox.start(repo, tmp_path.name.replace("_", "-")[-40:])
    yield box
    box.destroy()


@pytest.fixture
def run_report(retriever, sandbox_image, tmp_path):
    def _run(llm, on_update=None, **state):
        return asyncio.run(run_issue(
            FIXTURE_REPO, "Pagination skips the first page", llm=llm, model_name=FAKE_MODEL,
            retriever=retriever, workspace_root=str(tmp_path), on_update=on_update, **state,
        ))

    return _run


@pytest.fixture
def run(run_report):
    """Most tests only care about the final state."""
    return lambda llm, **state: run_report(llm, **state).state
