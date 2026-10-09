import hashlib
import re
import threading
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel

from retrieval.chunker import Chunk, chunk_file, language_for, skip_reason
from retrieval.embeddings import Embedder
from retrieval.store import ChunkStore, SearchFilters, StoredChunk

# Standard reciprocal rank fusion constant: damps the advantage of rank 1 so agreement across lists wins.
RRF_K = 60
# What gets embedded per chunk: the label plus the start of the code, about 600 tokens. Enough for a signature,
# docstring and the first part of the body; the full chunk is still stored and keyword-searchable. Embedding cost
# grows with the square of length, so uncapped chunks make indexing a large repo slow and memory-hungry.
MAX_EMBED_CHARS = 2000
# Embeddings are cached in slices this big, so a large first index shows progress and survives an interruption.
CACHE_WRITE_EVERY = 512

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "not", "but", "has", "have", "had",
    "instead", "when", "should", "would", "could", "there", "their", "into", "also", "than", "then", "them", "they",
    "returns", "return", "expected", "actual", "clearly", "comes", "back", "does", "like", "just", "what", "which",
}
_IDENTIFIER = re.compile(r"`([^`]+)`|\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+|[a-z0-9]+_[a-z0-9_]+|[a-z]+[A-Z][A-Za-z0-9]*|[A-Za-z_][A-Za-z0-9_]*(?=\())")


class ScoredChunk(StoredChunk):
    score: float


class IndexReport(BaseModel):
    chunks: int = 0
    # Chunks the skipped files would have produced, by reason (too_large, long_lines, generated, data).
    skipped_chunks: dict[str, int] = {}
    skipped_files: int = 0

    def merge(self, other: "IndexReport") -> "IndexReport":
        skipped = dict(self.skipped_chunks)
        for reason, n in other.skipped_chunks.items():
            skipped[reason] = skipped.get(reason, 0) + n
        return IndexReport(chunks=self.chunks + other.chunks, skipped_chunks=skipped,
                           skipped_files=self.skipped_files + other.skipped_files)


def chunk_paths(repo: Path, paths: list[str]) -> tuple[list[Chunk], IndexReport]:
    """Chunk files for indexing, leaving out noise and counting what it would have added."""
    chunks: list[Chunk] = []
    report = IndexReport()
    for path in paths:
        file = Path(repo) / path
        # Only files we would chunk are classified; .gitignore or a minified .js is not noise, just not indexed.
        if language_for(path) is None or not file.is_file():
            continue
        try:
            text = file.read_text()
        except UnicodeDecodeError:
            continue
        reason = skip_reason(path, text)
        if reason:
            report.skipped_chunks[reason] = report.skipped_chunks.get(reason, 0) + len(chunk_file(path, text, keep_noise=True))
            report.skipped_files += 1
            continue
        chunks.extend(chunk_file(path, text))
    report.chunks = len(chunks)
    return chunks, report


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
    def __init__(self, store: ChunkStore, embedder: Embedder, progress=None):
        self.store = store
        self.embedder = embedder
        self.progress = progress  # optional callable(done, total) while embedding uncached chunks
        # One embedding job at a time across concurrent runs: bounds memory, and a second run indexing the same repo
        # waits and then finds the first run's embeddings in the cache instead of computing them again.
        self._embed_lock = threading.Lock()

    def index_paths(self, index_id: str, repo: Path, paths: list[str]) -> IndexReport:
        """(Re)index the given files. Files that no longer exist are just removed from the index."""
        self.store.delete_paths(index_id, paths)
        chunks, report = chunk_paths(repo, paths)
        if chunks:
            # Path and symbol go into the embedded text, which helps when names carry the meaning. Line numbers do
            # not: they would make every chunk below an edit look new to the cache.
            texts = [f"{c.embedding_label()}\n{c.content}"[:MAX_EMBED_CHARS] for c in chunks]
            self.store.add(index_id, chunks, self._embed_with_cache(texts))
        return report

    def _embed_with_cache(self, texts: list[str]) -> list[list[float]]:
        with self._embed_lock:
            return self._embed_with_cache_unlocked(texts)

    def _embed_with_cache_unlocked(self, texts: list[str]) -> list[list[float]]:
        hashes = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
        cached = self.store.cached_embeddings(self.embedder.model_name, hashes)
        missing = list({h: t for h, t in zip(hashes, texts, strict=True) if h not in cached}.items())
        for start in range(0, len(missing), CACHE_WRITE_EVERY):
            batch = missing[start:start + CACHE_WRITE_EVERY]
            fresh = dict(zip([h for h, _ in batch], self.embedder.embed_documents([t for _, t in batch]), strict=True))
            self.store.cache_embeddings(self.embedder.model_name, fresh)
            cached.update(fresh)
            if self.progress:
                self.progress(min(start + CACHE_WRITE_EVERY, len(missing)), len(missing))
        return [cached[h] for h in hashes]

    def search(self, index_id: str, text: str, top_k: int = 20, filters: SearchFilters | None = None) -> list[ScoredChunk]:
        query = rewrite_query(text)
        pool = top_k * 3
        return fuse(
            [
                self.store.vector_search(index_id, self.embedder.embed_query(text[: MAX_EMBED_CHARS * 2]), pool, filters),
                self.store.keyword_search(index_id, query.keywords, pool, filters),
                self.store.symbol_search(index_id, query.identifiers, pool, filters),
            ],
            top_k,
        )

    def drop(self, index_id: str) -> None:
        self.store.drop_index(index_id)
