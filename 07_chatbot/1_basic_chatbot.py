from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: AgentState):
    return {
        "messages": [llm.invoke(state["messages"])]
    }

graph = StateGraph(AgentState)
graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)

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

    