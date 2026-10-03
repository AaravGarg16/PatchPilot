import os

import psycopg
import pytest

from retrieval.retriever import Retriever
from retrieval.store import ChunkStore
from tests.fakes import HashEmbedder

# A separate database so tests never touch the dev index. Created once with: createdb patchpilot_test
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "postgresql://localhost:5432/patchpilot_test")


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
