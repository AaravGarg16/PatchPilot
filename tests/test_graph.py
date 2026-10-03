from agent.graph import build_graph
from agent.state import PatchPilotState


def test_graph_runs_to_completion(capsys):
    result = build_graph().invoke(PatchPilotState(issue="smoke test"))

    assert result["issue"] == "smoke test"
    assert "PatchPilot graph is running" in capsys.readouterr().out
