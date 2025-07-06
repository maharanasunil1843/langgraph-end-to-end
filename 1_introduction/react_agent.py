from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, tool
from langchain_community.tools import TavilySearchResults
import datetime

from dotenv import load_dotenv
load_dotenv()

# Initialize the Google Generative AI model
llm_1 = ChatOpenAI(model="gpt-4.1-nano", temperature=0.5)

# result = llm_1.invoke("Give a tweet about today's weather in Hyderabad, India.")

# print(result)

# Initialize the Tavily search tool
tavily_search = TavilySearchResults(search_depth="basic")

# Define a custom tool to know the current date and time
@tool
def get_current_datetime(input: str) -> str:
    """Always use this tool to get the current date and time. Do not guess."""
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time

tools=[tavily_search, get_current_datetime]

agent = initialize_agent(
    tools=tools, 
    llm=llm_1, 
    agent="zero-shot-react-description", 
    verbose=True,
    handle_parsing_errors=True
    )
result = agent.invoke("When was SpaceX's last launch and how many days ago was that from this instant?")