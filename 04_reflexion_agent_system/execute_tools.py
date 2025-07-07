import json
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage
from langchain_tavily import TavilySearch

# Create the Tavily search tool
tavily_tool = TavilySearch(max_results=5)

# Function to execute search queries from ANswerQuestion tool calls
def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
    last_ai_message: AIMessage = state[-1]

    # Extract tool calls from the AI Message
    if not hasattr(last_ai_message, "tool_calls") or not last_ai_message.tool_calls:
        return []

    # Process the AnswerQuestion or ReviseAnswer tool calls to extract search queres.
    for tool_call in last_ai_message.tool_calls:
        if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
            call_id = tool_call["id"]
            search_queries = tool_call["args"].get("search_queries", [])

            # Execute each search query using the tavily tool
            query_results = {}
            for query in search_queries:
                result = tavily_tool.invoke(query)
                query_results[query] = result

            # Create a tool message with the results
            return [ToolMessage(
                    content=json.dumps(query_results),
                    tool_call_id=call_id
                )]
