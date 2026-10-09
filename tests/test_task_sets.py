import json

from evals.task_sets import allocate, build_sets
from evals.tasks import EVALS_DIR, suite_instance_ids


def rows(**per_repo: int) -> list[dict]:
    return [{"instance_id": f"{repo}__{repo}-{i}", "repo": f"{repo}/{repo}"}
            for repo, n in per_repo.items() for i in range(n)]


def test_largest_remainder_allocation():
    assert allocate({"a": 114, "b": 77}, 8) == {"a": 5, "b": 3}
    assert sum(allocate({"a": 5, "b": 3, "c": 1}, 7).values()) == 7
    assert allocate({"a": 1, "b": 1}, 1) == {"a": 1}  # ties go by name, never by dict order


def test_sets_are_seeded_reproducible_and_share_draws():
    data = rows(django=40, sympy=30, pytest=10, flask=2)
    first = build_sets(data, dev_ids=set())
    assert first == build_sets(list(reversed(data)), dev_ids=set())  # input order does not matter
    ablation, lite = first["ablation8"]["instance_ids"], first["lite20"]["instance_ids"]
    assert len(ablation) == 8 and len(lite) == 20 and len(set(lite)) == 20
    # Shared by construction: each repo's order is fixed, and every set takes a prefix of it, so lite20's larger
    # django and sympy samples contain ablation8's.
    assert set(ablation) <= set(lite)
    assert build_sets(data, dev_ids=set(), seed=1)["lite20"]["instance_ids"] != lite  # the seed matters


def test_dev_tasks_are_never_drawn():
    data = rows(django=12, sympy=8)
    dev = {f"django__django-{i}" for i in range(6)}
    sets = build_sets(data, dev_ids=dev)
    assert not dev & set(sets["ablation8"]["instance_ids"] + sets["lite20"]["instance_ids"])


def test_committed_sets_are_consistent():
    smoke = set(suite_instance_ids("smoke"))
    for name, size in (("ablation8", 8), ("lite20", 20)):
        spec = json.loads((EVALS_DIR / "tasks" / f"{name}.json").read_text())
        ids = spec["instance_ids"]
        assert len(ids) == len(set(ids)) == size == sum(spec["allocation"].values())
        assert not smoke & set(ids)
        for repo, n in spec["allocation"].items():
            assert sum(i.startswith(repo.replace("/", "__") + "-") for i in ids) == n
    assert set(suite_instance_ids("ablation8")) <= set(suite_instance_ids("lite20"))
