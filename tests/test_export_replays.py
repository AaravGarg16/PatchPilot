"""The replay UI's data: only resolved runs, and exactly what the result files recorded."""
import json

import pytest

import evals.export_replays as export
from evals.results import Grade, TaskResult
from evals.tasks import PROJECT_ROOT

ISSUE = {"repo": "owner/repo", "version": "1.0", "problem_statement": "Widget breaks\nIt breaks when used.\n"}
PATCH = "diff --git a/pkg/widget.py b/pkg/widget.py\n--- a/pkg/widget.py\n+++ b/pkg/widget.py\n@@ -1 +1 @@\n-x\n+y\n"


def save(tmp_path, monkeypatch, **overrides) -> None:
    fields = {
        "instance_id": "owner__repo-1", "source": "swebench", "config": "full", "status": "done", "attempts": 2,
        "self_accepted": True, "tests_after_passed": True, "model_patch": PATCH, "cost_usd": 0.5, "latency_seconds": 61.4,
        "retrieved_files": ["pkg/widget.py", "docs/a.txt"], "planned_files": ["pkg/widget.py"],
        "rejections": [{"attempt": 1, "tests_ran": True, "tests_passed": False, "tests_selected": ["tests/test_widget.py"],
                        "problems": ["Still wrong."]}],
        "final_evaluation": {"correct": True, "requirements_met": True, "tests_pass": True, "problems": [],
                             "recommended_action": "finish"},
        "grade": Grade(grader="swebench", resolved=True, patch_applied=True, fail_to_pass_passed=1, fail_to_pass_total=1,
                       pass_to_pass_passed=4, pass_to_pass_total=4),
    }
    result = TaskResult(**{**fields, **overrides})
    path = tmp_path / "results" / "run" / "full" / "tasks" / "owner__repo-1.json"
    path.parent.mkdir(parents=True)
    path.write_text(result.model_dump_json())
    monkeypatch.setattr(export, "EVALS_DIR", tmp_path)


def build():
    return export.build_replay("run", "full", "owner__repo-1", "Tag", "Takeaway.", {"owner__repo-1": ISSUE})


def test_replay_carries_only_recorded_facts(tmp_path, monkeypatch):
    save(tmp_path, monkeypatch)
    replay = build()
    assert replay["issue"] == {"title": "Widget breaks", "body": "It breaks when used."}
    assert replay["retrieval"] == [{"path": "pkg/widget.py", "inPatch": True}, {"path": "docs/a.txt", "inPatch": False}]
    first, second = replay["attempts"]
    assert (first["accepted"], first["tests"]["selected"], first["problems"]) == (False, ["tests/test_widget.py"], ["Still wrong."])
    # The accepted attempt: which test files ran was not recorded, and the export says so rather than guessing.
    assert (second["number"], second["accepted"], second["tests"]) == (2, True, {"ran": True, "passed": True, "selected": None})
    assert replay["patch"] == PATCH and replay["grade"] == {"resolved": True, "failToPass": [1, 1], "passToPass": [4, 4]}
    assert replay["metrics"]["latencySeconds"] == 61 and replay["metrics"]["costUsd"] == 0.5


def test_a_run_the_official_grader_did_not_resolve_cannot_be_featured(tmp_path, monkeypatch):
    save(tmp_path, monkeypatch, grade=Grade(grader="swebench", resolved=False, patch_applied=True))
    with pytest.raises(ValueError, match="not resolved by the official grader"):
        build()


def test_bundled_replays_match_their_result_files():
    data = json.loads((PROJECT_ROOT / "apps" / "web" / "src" / "data" / "replays.json").read_text())
    assert 2 <= len(data["replays"]) <= 3
    for replay in data["replays"]:
        source = TaskResult.model_validate_json((PROJECT_ROOT / replay["source"]).read_text())
        assert source.grade.resolved and replay["grade"]["resolved"]
        assert replay["patch"] == source.model_patch
        assert replay["selfAccepted"] == source.self_accepted
        assert replay["metrics"]["attempts"] == source.attempts and replay["metrics"]["toolCalls"] == source.tool_calls
        assert replay["metrics"]["costUsd"] == pytest.approx(source.cost_usd, abs=1e-4)
        assert [a["problems"] for a in replay["attempts"] if not a["accepted"]] == [r["problems"] for r in source.rejections]
