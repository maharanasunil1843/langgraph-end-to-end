from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, START, END
import operator

class SimpleState(TypedDict):
    count: int
    sum: Annotated[int, operator.add]
    history: Annotated[List[int], operator.concat]

def increment(state: SimpleState) -> SimpleState:
    new_count = state["count"] + 1
    return {
        "count": new_count,
        "sum" : new_count,
        "history": [new_count]
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

# print(app.get_graph().draw_mermaid())

response = app.invoke({
    "count" : 0,
    "sum": 0,
    "history": []
})

print(response)

