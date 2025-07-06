from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

generation_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a twitter technie influencer assistant tasked with writing excellent posts."
     "Generate the best twitter post possible for the user's request."
     "If the user provides critique, respond with a better and refined version of the post."),
    MessagesPlaceholder(variable_name="messages")
])

reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
     "Always provide detailed recommendations, including request for length, virality, style ,etc."),
    MessagesPlaceholder(variable_name="messages"),    
])

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.5)

generate_tweet_chain = generation_prompt | llm
reflect_tweet_chain = reflection_prompt | llm