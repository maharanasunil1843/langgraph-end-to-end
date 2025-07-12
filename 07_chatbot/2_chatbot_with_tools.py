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

response = llm_with_tools.invoke("What's the weather in Hyderabad?")

print(response)

