import uuid

import pytest

from retrieval.retriever import ScoredChunk, fuse, rewrite_query, select_files
from retrieval.store import ChunkStore, SearchFilters, StoredChunk
from tests.fakes import FIXED_PAGINATION, FIXTURE_REPO
from tools import filesystem, git

ISSUE = "Pagination skips the first page: `paginate(orders, 1, 10)` is off by one and total_pages() rounds down."


@pytest.fixture
def indexed(retriever, tmp_path):
    repo = git.checkout_repo(FIXTURE_REPO, tmp_path / "repo")
    index_id = tmp_path.name
    retriever.index_paths(index_id, repo, git.list_files(repo))
    yield retriever, repo, index_id
    retriever.drop(index_id)


def stored(id_: int, path: str) -> StoredChunk:
    return StoredChunk(id=id_, path=path, language="python", symbol=None, chunk_type="function",
                       start_line=1, end_line=2, content="")


def test_rewrite_query_pulls_out_identifiers_and_keywords():
    query = rewrite_query(ISSUE)
    assert {"paginate", "total_pages"} <= set(query.identifiers)
    assert "pagination" in query.keywords
    assert "the" not in query.keywords and "10" not in query.keywords


def test_fusion_rewards_agreement_across_lists():
    a, b, c = stored(1, "a.py"), stored(2, "b.py"), stored(3, "c.py")
    fused = fuse([[a, b, c], [b, c], [b]], top_k=2)
    assert [chunk.id for chunk in fused] == [2, 3]


def test_select_files_ranks_by_best_chunk_not_chunk_count():
    def scored(path, score):
        return ScoredChunk(**stored(0, path).model_dump(), score=score)

    chunks = [scored("src.py", 0.05), *(scored("tests.py", 0.02) for _ in range(5))]
    assert select_files(chunks, 2) == ["src.py", "tests.py"]


def test_hybrid_search_finds_the_buggy_code_first(indexed):
    retriever, _, index_id = indexed
    hits = retriever.search(index_id, ISSUE, top_k=5)
    assert {hits[0].symbol, hits[1].symbol} == {"paginate", "total_pages"}
    assert select_files(hits, 3)[0] == "shop/pagination.py"


def test_symbol_search_matches_exact_names(indexed):
    retriever, _, index_id = indexed
    hits = retriever.store.symbol_search(index_id, ["format_price"], limit=5)
    assert [(h.path, h.symbol) for h in hits] == [("shop/formatting.py", "format_price")]


def test_metadata_filters_narrow_results(indexed):
    retriever, _, index_id = indexed
    docs = retriever.search(index_id, ISSUE, filters=SearchFilters(languages=["markdown"]))
    assert {h.path for h in docs} == {"README.md"}
    tests_only = retriever.search(index_id, ISSUE, filters=SearchFilters(path_prefix="tests/"))
    assert tests_only and all(h.path.startswith("tests/") for h in tests_only)


def test_reindexing_replaces_stale_chunks_and_drops_deleted_files(indexed):
    retriever, repo, index_id = indexed
    filesystem.write_file(repo, "shop/pagination.py", FIXED_PAGINATION)
    (repo / "shop" / "formatting.py").unlink()

    retriever.index_paths(index_id, repo, ["shop/pagination.py", "shop/formatting.py"])

    (paginate,) = retriever.store.symbol_search(index_id, ["paginate"], limit=5)
    assert "(page - 1) * per_page" in paginate.content
    assert retriever.store.symbol_search(index_id, ["format_price"], limit=5) == []


def test_runs_are_isolated_by_index_id(indexed):
    retriever, _, index_id = indexed
    assert retriever.search("some-other-run", ISSUE) == []
    retriever.drop(index_id)
    assert retriever.store.count(index_id) == 0


def test_changing_embedding_dimension_recreates_the_table(test_db_url):
    ChunkStore(test_db_url, 8)
    store = ChunkStore(test_db_url, 256)  # back to the HashEmbedder size for the other tests
    assert store.count("anything") == 0


def test_embedded_text_is_capped_but_stored_content_is_whole(retriever, tmp_path):
    from retrieval.retriever import MAX_EMBED_CHARS

    seen: list[str] = []
    original = retriever.embedder.embed_documents
    retriever.embedder.embed_documents = lambda texts: seen.extend(texts) or original(texts)
    repo = tmp_path / "repo"
    body = "\n".join(f"    value_{i} = '{'x' * 40}'" for i in range(110))
    # Unique per run: the embedding cache persists across test runs, and a cached chunk is never re-embedded.
    filesystem.write_file(repo, "big.py", f"def big():\n    run = '{uuid.uuid4().hex}'\n{body}\n")

    retriever.index_paths("cap-test", repo, ["big.py"])
    try:
        (stored,) = retriever.store.symbol_search("cap-test", ["big"], limit=5)
    finally:
        retriever.drop("cap-test")
    assert len(stored.content) > MAX_EMBED_CHARS
    assert seen and all(len(t) <= MAX_EMBED_CHARS for t in seen)


def test_embeddings_are_cached_in_slices_with_progress(retriever, tmp_path, monkeypatch):
    import retrieval.retriever as retriever_module

    monkeypatch.setattr(retriever_module, "CACHE_WRITE_EVERY", 2)
    seen: list[tuple[int, int]] = []
    retriever.progress = lambda done, total: seen.append((done, total))
    repo = tmp_path / "repo"
    run = uuid.uuid4().hex  # unique per run: the embedding cache persists across test runs
    for i in range(5):
        filesystem.write_file(repo, f"m{i}.py", f"def fn_{i}_{run}():\n    return {i}\n")

    retriever.index_paths("slices", repo, [f"m{i}.py" for i in range(5)])
    retriever.drop("slices")
    assert seen == [(2, 5), (4, 5), (5, 5)]


def test_only_indexable_files_are_counted_as_skipped(tmp_path):
    from retrieval.retriever import chunk_paths

    repo = tmp_path / "repo"
    filesystem.write_file(repo, ".gitignore", "# generated by a tool\n*.pyc\n")
    filesystem.write_file(repo, "static/app.min.js", "x" * 5000)
    filesystem.write_file(repo, "tests/fixtures/users.json", '{"id": 1}\n')
    filesystem.write_file(repo, "app/models.py", "class User: pass\n")

    chunks, report = chunk_paths(repo, [".gitignore", "static/app.min.js", "tests/fixtures/users.json", "app/models.py"])
    assert [c.path for c in chunks] == ["app/models.py"]
    assert report.skipped_chunks == {"data": 1} and report.skipped_files == 1


def test_code_that_only_moved_is_not_re_embedded(retriever, tmp_path):
    function = f"def moved_{uuid.uuid4().hex}():\n    return 42\n"
    repo = tmp_path / "repo"
    filesystem.write_file(repo, "m.py", function)
    retriever.index_paths("moved", repo, ["m.py"])

    embedded: list[str] = []
    original = retriever.embedder.embed_documents
    retriever.embedder.embed_documents = lambda texts: embedded.extend(texts) or original(texts)
    filesystem.write_file(repo, "m.py", "import os\n\n\n" + function)  # same function, three lines lower
    try:
        retriever.index_paths("moved", repo, ["m.py"])
        stored = retriever.store.symbol_search("moved", [function.split("(")[0][4:]], limit=5)
    finally:
        retriever.drop("moved")
    assert stored and stored[0].start_line == 4  # the stored chunk still has its real, new position
    assert all("return 42" not in text for text in embedded)  # ...but its embedding came from the cache
