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
    # Check if there are tool results in the conversation
    has_tool_results = any(
        hasattr(msg, 'content') and msg.content and '>' in msg.content 
        for msg in state["messages"] 
        if hasattr(msg, 'content')
    )
    
    if has_tool_results:
        # If we have tool results, instruct the LLM to use them
        messages = state["messages"].copy()
        system_msg = HumanMessage(content="""You are a helpful assistant. The user has asked a question and search results have been provided. 
        Please use these search results to provide a comprehensive and accurate answer to the user's question. 
        If the search results contain relevant information, use it. If not, politely explain that you don't have the specific information.""")
        messages.insert(0, system_msg)
        return {
            "messages": [llm.invoke(messages)]
        }
    else:
        # Normal tool call generation
        return {
            "messages": [llm_with_tools.invoke(state["messages"])]
        }

def tools_router(state: AgentState):
    last_message = state["messages"][-1]
    if (hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
        return "tool_node"
    else:
        return "summarize"

def should_continue_after_tools(state: AgentState):
    last_message = state["messages"][-1]
    if (hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
        return "chatbot"  # More tools needed
    else:
        return "summarize"  # No more tools, go to summarize
    
def summarize(state: AgentState):
    # Create a better prompt that instructs the LLM to use the search results
    messages = state["messages"].copy()
    
    # Add a system message to instruct the LLM to use the search results
    system_message = HumanMessage(content="""You are a helpful assistant. Use the search results provided in the conversation to answer the user's question. 
    If search results are available, use them to provide a comprehensive and accurate answer. 
    If no search results are available, politely explain that you don't have the information.""")
    
    # Add the system message at the beginning
    messages.insert(0, system_message)
    
    summary = llm.invoke(messages)
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
graph.add_conditional_edges("tool_node", should_continue_after_tools)
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
