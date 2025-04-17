from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent,AgentExecutor


load_dotenv()


class ResearchResponse(BaseModel):
    topic:str
    summary:str
    sources:list[str]
    tools_used:list[str]


llm=ChatOpenAI(model="gpt-4o-mini")

parser = PydanticOutputParser(pydantic_object=ResearchResponse)


prompt = ChatPromptTemplate.from_messages(
   [
    (
    "system",
    """
    You are an expert researcher.
    
    Provide a short summary, relevant sources, and tools used for the research.
    
    Respond in the following JSON format:
    {format_instructions}
    """,
    ),
    ("placeholder","{chat_history}"),
    ("human","{query}"),
    ("placeholder","{agent_scratchpad}"),
   ]
).partial(format_instructions=parser.get_format_instructions())


agent=create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=[]
)


agent_executor=AgentExecutor(agent=agent,tools=[],verbose=True)
raw_response=agent_executor.invoke({"query":"What is the meaning of arjun?"})
print(raw_response)




# response=llm.invoke("what is the meaning of arjun")
# print(response)