from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class SimpleState(TypedDict):
    count: int

def increment(state: SimpleState) -> SimpleState:
    return {
        "count": state["count"] + 1
    }

def should_continue(state: SimpleState) -> str:
    if state["count"] < 5:
        return "continue"
    else:
        return "stop"

graph = StateGraph(SimpleState)

graph.add_node("increment", increment)
graph.add_edge(START, "increment")
graph.add_conditional_edges(
    "increment",
    should_continue,
    {
        "continue": "increment",
        "stop": END
    }
)

app = graph.compile()

print(app.get_graph().draw_mermaid())

response = app.invoke({
    "count" : 0
})

print(response)

