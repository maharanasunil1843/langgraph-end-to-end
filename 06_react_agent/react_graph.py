from dotenv import load_dotenv
load_dotenv()

from langchain_core.agents import AgentAction, AgentFinish
from langgraph.graph import END, StateGraph

from nodes import reason_node, act_node
from react_state import AgentState

def should_continue(state: AgentState) -> str:
    if isinstance(state["agent_outcome"], AgentFinish):
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)

graph.add_node("reason_node", reason_node)
graph.add_node("act_node", act_node)
graph.set_entry_point("reason_node")

graph.add_conditional_edges(
    "reason_node",
    should_continue,
    {
        "end": END,
        "continue": "act_node"
    }
)

graph.add_edge("act_node", "reason_node")

app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({
        "input": "How many days ago was the latest SpaceX launch?",
        "agent_outcome": None,
        "intermediate_steps": []
    })
    print(result)
    print(result['agent_outcome'].return_values['output'], 'final_result')