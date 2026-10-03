from langgraph.graph import END, START, StateGraph

from agent.state import PatchPilotState


def hello(state: PatchPilotState) -> dict:
    print("PatchPilot graph is running")
    return {}


def build_graph():
    graph = StateGraph(PatchPilotState)
    graph.add_node("hello", hello)
    graph.add_edge(START, "hello")
    graph.add_edge("hello", END)
    return graph.compile()


def main():
    build_graph().invoke(PatchPilotState())


if __name__ == "__main__":
    main()
