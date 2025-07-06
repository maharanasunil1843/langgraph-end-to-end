from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import HumanMessage
from schema import AnswerQuestion
from dotenv import load_dotenv
import datetime

load_dotenv()
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0.5
)

pydantic_tools_parser = PydanticToolsParser(tools=[AnswerQuestion])

actor_agent_prompt = ChatPromptTemplate.from_messages(
    [
    (
        "system",
        """You are an expert AI reshearcher and developer.
        Current time is {time}.
        
        1. {first_instruction}
        2. Reflect and critique your answer. Be severe to maximize improvement.
        3. After the reflection, **list 1-3 search queries separately** for researching improvements. Do not include them inside the reflection.
        """,
    ),
    MessagesPlaceholder(variable_name="messages"),
    (
        "system",
        """Answer the user's question above using the required format."""
    ),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_agent_prompt.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

first_responder_chain = first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion") | pydantic_tools_parser

response = first_responder_chain.invoke({
    "messages": [
        HumanMessage(
            content="Write a blog post on how small business can leverage AI to grow."
        )
    ]
})

print(response)

