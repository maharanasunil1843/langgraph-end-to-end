from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.messages import HumanMessage
from schema import AnswerQuestion, ReviseAnswer
from dotenv import load_dotenv
import datetime

load_dotenv()
llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.5
)

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

first_responder_chain = first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")

validator = PydanticToolsParser(tools=[AnswerQuestion])

revise_instructions = """
Revise your previous answer using the new information.
- You should use the previous critique to add important information to your answer.
- You MUST include numerical citations in your revised answer to ensure it can be verified.
- You MUST add a "References" section to the bottom of your answer (which does not count towards th world limit) in form of following patterns:
    - [1] https://example.com
    - [2] https://example.com
- You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor_chain = actor_agent_prompt.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

if __name__ == "__main__":
    response = first_responder_chain.invoke({
    "messages": [
        HumanMessage(
            content="Write a blog post on how to get free from procrastination."
        )
    ]
})
    print(response)

