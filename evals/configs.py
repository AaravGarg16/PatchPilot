"""Named agent configurations and the experiments that compare them."""
from agent.state import AgentConfig

CONFIGS: dict[str, AgentConfig] = {
    "baseline": AgentConfig(rag=False, planner=False, evaluator=False),
    "rag": AgentConfig(rag=True, planner=False, evaluator=False),
    "planner": AgentConfig(rag=True, planner=True, evaluator=False),
    "full": AgentConfig(),
    "full-history": AgentConfig(context_mode="full_history"),
}

LABELS = {
    "baseline": "Baseline (executor + tests)",
    "rag": "+ RAG",
    "planner": "+ Planner",
    "full": "+ Evaluator (full PatchPilot)",
    "full-history": "Full history (no context engineering)",
}

EXPERIMENTS = {
    # Each row adds one component to the one above.
    "ablation": ["baseline", "rag", "planner", "full"],
    # Same components; only how context is managed differs.
    "context": ["full", "full-history"],
}
