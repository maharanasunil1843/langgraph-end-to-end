from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode
import json
load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    
search_tool = TavilySearch(max_results=2)
tools=[search_tool]
llm=ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: AgentState):
    return {
        "messages": [llm_with_tools.invoke(state["messages"])]
    }

def tools_router(state: AgentState):
    last_message = state["messages"][-1]
    if (hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
        return "tool_node"
    else:
        return "summarize"
    
def summarize(state: AgentState):
    summary = llm.invoke(state["messages"])
    return {
        "messages": state["messages"] + [AIMessage(content=summary.content)]
    }
    
tool_node = ToolNode(tools=tools)

graph = StateGraph(AgentState)

graph.add_node("chatbot", chatbot)
graph.add_node("tool_node", tool_node)
graph.add_node("summarize", summarize)
graph.set_entry_point("chatbot")
graph.add_conditional_edges("chatbot", tools_router)
graph.add_edge("tool_node", "summarize")
graph.add_edge("summarize", END)

app = graph.compile()


while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit", "end"]:
        break
    else:
        response = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        })
        print(response)
