import re
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel

from retrieval.chunker import MAX_FILE_BYTES, chunk_file
from retrieval.embeddings import Embedder
from retrieval.store import ChunkStore, SearchFilters, StoredChunk

# Standard reciprocal rank fusion constant: damps the advantage of rank 1 so agreement across lists wins.
RRF_K = 60

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "not", "but", "has", "have", "had",
    "instead", "when", "should", "would", "could", "there", "their", "into", "also", "than", "then", "them", "they",
    "returns", "return", "expected", "actual", "clearly", "comes", "back", "does", "like", "just", "what", "which",
}
_IDENTIFIER = re.compile(r"`([^`]+)`|\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+|[a-z0-9]+_[a-z0-9_]+|[a-z]+[A-Z][A-Za-z0-9]*|[A-Za-z_][A-Za-z0-9_]*(?=\())")


class ScoredChunk(StoredChunk):
    score: float


class RetrievalQuery(BaseModel):
    text: str
    keywords: list[str]
    identifiers: list[str]


def rewrite_query(text: str) -> RetrievalQuery:
    """Turn an issue into search inputs without an LLM call: code identifiers for symbol lookup, words for keyword search."""
    identifiers: list[str] = []
    for backticked, bare in _IDENTIFIER.findall(text):
        candidates = re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", backticked) if backticked else [bare]
        for name in candidates:
            for part in [name, name.split(".")[-1]]:
                if part not in identifiers and len(part) > 2:
                    identifiers.append(part)
    words = re.findall(r"[a-z0-9_]+", text.lower())
    keywords = list(dict.fromkeys(w for w in words if len(w) > 2 and w not in _STOPWORDS and not w.isdigit()))
    return RetrievalQuery(text=text, keywords=keywords, identifiers=identifiers)


def fuse(ranked_lists: list[list[StoredChunk]], top_k: int) -> list[ScoredChunk]:
    scores: dict[int, float] = defaultdict(float)
    by_id: dict[int, StoredChunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.id] += 1 / (RRF_K + rank)
            by_id[chunk.id] = chunk
    best = sorted(scores, key=lambda i: (-scores[i], i))[:top_k]
    return [ScoredChunk(**by_id[i].model_dump(), score=scores[i]) for i in best]


def select_files(chunks: list[ScoredChunk], max_files: int) -> list[str]:
    """Rank files by their best chunk; summed score only breaks ties, so files with many weak chunks don't win."""
    best: dict[str, float] = defaultdict(float)
    totals: dict[str, float] = defaultdict(float)
    for chunk in chunks:
        best[chunk.path] = max(best[chunk.path], chunk.score)
        totals[chunk.path] += chunk.score
    return sorted(best, key=lambda p: (-best[p], -totals[p]))[:max_files]


class Retriever:
    def __init__(self, store: ChunkStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def index_paths(self, index_id: str, repo: Path, paths: list[str]) -> int:
        """(Re)index the given files. Files that no longer exist are just removed from the index."""
        self.store.delete_paths(index_id, paths)
        chunks = []
        for path in paths:
            file = Path(repo) / path
            if file.is_file() and file.stat().st_size <= MAX_FILE_BYTES:
                try:
                    chunks.extend(chunk_file(path, file.read_text()))
                except UnicodeDecodeError:
                    continue
        if chunks:
            # The label puts path and symbol into the embedded text, which helps when names carry the meaning.
            embeddings = self.embedder.embed_documents([f"{c.label()}\n{c.content}" for c in chunks])
            self.store.add(index_id, chunks, embeddings)
        return len(chunks)

    def search(self, index_id: str, text: str, top_k: int = 20, filters: SearchFilters | None = None) -> list[ScoredChunk]:
        query = rewrite_query(text)
        pool = top_k * 3
        return fuse(
            [
                self.store.vector_search(index_id, self.embedder.embed_query(text), pool, filters),
                self.store.keyword_search(index_id, query.keywords, pool, filters),
                self.store.symbol_search(index_id, query.identifiers, pool, filters),
            ],
            top_k,
        )

    def drop(self, index_id: str) -> None:
        self.store.drop_index(index_id)
