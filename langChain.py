from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from tools import search_tool, wiki_tool

llm = OllamaLLM(model="llama3.1:8b")

class ResearchResponse(BaseModel):
  topic: str
  summary: str
  sources: list[str]
  tools_used: list[str]

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
  [
    (
      "system",
      """
      You are a research assistant that will help generate a research paper.
      Answer the user query and use necessary tools.
      Wrap the output in this format and provide no other text\n{format_instructions}
      """,
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
  ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool]
agent = initialize_agent(
  llm=llm,
  tools=tools,
  agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
  verbose=True,
)

query = input("Enter your research query: ")
raw_response = agent.run(query)

try:
  structured_response = parser.parse(raw_response)
  print(structured_response)
except Exception as e:
  print("Failed to parse response:", e, "Raw response:", raw_response)