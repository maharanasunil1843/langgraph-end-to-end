from typing import List, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from chains import generate_tweet_chain, reflect_tweet_chain
from langgraph.graph import MessageGraph, END

load_dotenv()

graph = MessageGraph()


def generate_node(state):
    response = generate_tweet_chain.invoke({
        "messages": state
    })
    return state + [AIMessage(content=response.content)]

def reflect_node(state):
    response = reflect_tweet_chain.invoke({
        "messages": state
    })
    return state + [HumanMessage(content=response.content)]

def should_continue(state):
    if (len(state) > 6):
        return END
    return "reflect"

graph.add_node("generate", generate_node)
graph.add_node("reflect", reflect_node)

graph.set_entry_point("generate")

graph.add_conditional_edges(
    "generate",
    should_continue
)

graph.add_edge("reflect", "generate")

app = graph.compile()

# print(app.get_graph().draw_mermaid())
# print(app.get_graph().draw_ascii())

# print("Nodes:", app.get_graph().nodes)
# print("Edges:", app.get_graph().edges)

# # Test the graph executing to verify the loop
# initial_state = [HumanMessage(content="Write a tweet about AI.")]
# for step in app.stream(initial_state):
#     print("Step:", step)
#     for node, state in step.items():
#         print(f"Node: {node}, State: {state}")
#     print("--------")

response = app.invoke([HumanMessage(content="Write a tweet about the Gold Price's prediction for next 6 months in India. Keep the tweet comprehensive and engaging with atleast 200 words")])  
print(response)
  