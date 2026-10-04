import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from pydantic import BaseModel

from retrieval.chunker import Chunk

DEFAULT_DATABASE_URL = "postgresql://localhost:5432/patchpilot"

_COLUMNS = "id, path, language, symbol, chunk_type, start_line, end_line, content"


class SearchFilters(BaseModel):
    languages: list[str] | None = None
    path_prefix: str | None = None


class StoredChunk(Chunk):
    id: int


def database_url() -> str:
    load_dotenv()
    return os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL


class ChunkStore:
    """Code chunks for one or more indexed workspaces, with pgvector embeddings and a full-text column.

    Rows are scoped by index_id (one per agent run), so concurrent runs and old runs never mix.
    """

    def __init__(self, url: str, dim: int):
        self.url = url
        self.dim = dim
        self._ensure_schema()

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self.url, autocommit=True)
        register_vector(conn)
        # With a WHERE filter, a plain HNSW scan can return fewer than LIMIT rows; iterative scan keeps going.
        conn.execute("SET hnsw.iterative_scan = relaxed_order")
        return conn

    def _ensure_schema(self) -> None:
        with psycopg.connect(self.url, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            row = conn.execute(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = to_regclass('code_chunks') AND attname = 'embedding'"
            ).fetchone()
            if row and row[0] != self.dim:
                # Indexes are rebuilt every run, so switching embedding models just means starting the table over.
                conn.execute("DROP TABLE code_chunks")
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS code_chunks (
                    id bigserial PRIMARY KEY,
                    index_id text NOT NULL,
                    path text NOT NULL,
                    language text NOT NULL,
                    symbol text,
                    chunk_type text NOT NULL,
                    start_line int NOT NULL,
                    end_line int NOT NULL,
                    content text NOT NULL,
                    embedding vector({self.dim}) NOT NULL,
                    tsv tsvector GENERATED ALWAYS AS (
                        to_tsvector('simple', coalesce(symbol, '') || ' ' || path || ' ' || content)
                    ) STORED
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS code_chunks_index_path ON code_chunks (index_id, path)")
            conn.execute("CREATE INDEX IF NOT EXISTS code_chunks_tsv ON code_chunks USING gin (tsv)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS code_chunks_embedding ON code_chunks USING hnsw (embedding vector_cosine_ops)"
            )
            # Embeddings outlive runs: the same file at the same content embeds once, across tasks and eval configs.
            # Untyped vector, since different models have different dimensions; it is looked up by key, never searched.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    model text NOT NULL,
                    content_hash text NOT NULL,
                    embedding vector NOT NULL,
                    PRIMARY KEY (model, content_hash)
                )
            """)

    def add(self, index_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        rows = [
            (index_id, c.path, c.language, c.symbol, c.chunk_type, c.start_line, c.end_line, c.content, e)
            for c, e in zip(chunks, embeddings, strict=True)
        ]
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO code_chunks (index_id, path, language, symbol, chunk_type, start_line, end_line, content, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)",
                rows,
            )

    def delete_paths(self, index_id: str, paths: list[str]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM code_chunks WHERE index_id = %s AND path = ANY(%s)", (index_id, paths))

    def drop_index(self, index_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM code_chunks WHERE index_id = %s", (index_id,))

    def count(self, index_id: str) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT count(*) FROM code_chunks WHERE index_id = %s", (index_id,)).fetchone()[0]

    def _where(self, index_id: str, filters: SearchFilters | None) -> tuple[str, list]:
        clauses, params = ["index_id = %s"], [index_id]
        if filters and filters.languages:
            clauses.append("language = ANY(%s)")
            params.append(filters.languages)
        if filters and filters.path_prefix:
            clauses.append("path LIKE %s")
            params.append(filters.path_prefix.replace("%", r"\%").replace("_", r"\_") + "%")
        return " AND ".join(clauses), params

    def _query(self, sql: str, params: list) -> list[StoredChunk]:
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        keys = [k.strip() for k in _COLUMNS.split(",")]
        return [StoredChunk(**dict(zip(keys, row, strict=True))) for row in rows]

    def vector_search(self, index_id: str, vector: list[float], limit: int, filters: SearchFilters | None = None):
        where, params = self._where(index_id, filters)
        return self._query(
            f"SELECT {_COLUMNS} FROM code_chunks WHERE {where} ORDER BY embedding <=> %s::vector LIMIT %s",
            [*params, vector, limit],
        )

    def keyword_search(self, index_id: str, terms: list[str], limit: int, filters: SearchFilters | None = None):
        if not terms:
            return []
        where, params = self._where(index_id, filters)
        # Terms are pre-sanitized to [a-z0-9_]; OR them so any overlap counts, and let ts_rank_cd order by density.
        query = " | ".join(terms)
        return self._query(
            f"SELECT {_COLUMNS} FROM code_chunks, to_tsquery('simple', %s) q "
            f"WHERE {where} AND tsv @@ q ORDER BY ts_rank_cd(tsv, q) DESC, id LIMIT %s",
            [query, *params, limit],
        )

    def symbol_search(self, index_id: str, names: list[str], limit: int, filters: SearchFilters | None = None):
        if not names:
            return []
        where, params = self._where(index_id, filters)
        return self._query(
            f"SELECT {_COLUMNS} FROM code_chunks WHERE {where} "
            f"AND (symbol = ANY(%s) OR split_part(symbol, '.', -1) = ANY(%s)) ORDER BY id LIMIT %s",
            [*params, names, names, limit],
        )

    def cached_embeddings(self, model: str, hashes: list[str]) -> dict[str, list[float]]:
        if not hashes:
            return {}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT content_hash, embedding FROM embedding_cache WHERE model = %s AND content_hash = ANY(%s)",
                (model, hashes),
            ).fetchall()
        return {h: e.to_list() for h, e in rows}

    def cache_embeddings(self, model: str, items: dict[str, list[float]]) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO embedding_cache (model, content_hash, embedding) VALUES (%s, %s, %s::vector) "
                "ON CONFLICT DO NOTHING",
                [(model, h, e) for h, e in items.items()],
            )
