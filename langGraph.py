from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, START, END
from tools import search_tool

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
            Respond ONLY with a JSON object matching this schema, with example values filled in, and no explanations:
            {format_instructions}
            """,
        ),
        ("ai", "{chat_history}"),
        ("human", "{query}"),
        ("ai", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Define the state for LangGraph
class AgentState(BaseModel):
    query: str
    llm_output: str = ""
    parsed_output: ResearchResponse | None = None

# Node: LLM agent step
def agent_node(state: AgentState) -> AgentState:
    # Compose the prompt and run the LLM
    chat_history = ""
    agent_scratchpad = ""
    prompt_str = prompt.format(
        chat_history=chat_history,
        query=state.query,
        agent_scratchpad=agent_scratchpad,
    )
    llm_output = llm.invoke(prompt_str)
    return AgentState(query=state.query, llm_output=llm_output)

# Node: Tool use (search)
def tool_node(state: AgentState) -> AgentState:
    # For demo, just pass through (expand for multi-tool use)
    return state

# Node: Parse output
def parse_node(state: AgentState) -> AgentState:
    try:
        parsed = parser.parse(state.llm_output)
    except Exception as e:
        print("Failed to parse response:", e, "Raw response:", state.llm_output)
        parsed = None
    return AgentState(query=state.query, llm_output=state.llm_output, parsed_output=parsed)

# Build the LangGraph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tool", tool_node)
graph.add_node("parse", parse_node)

# Define the flow: START → agent → tool → parse → END
graph.add_edge(START, "agent")
graph.add_edge("agent", "tool")
graph.add_edge("tool", "parse")
graph.add_edge("parse", END)

app = graph.compile()

if __name__ == "__main__":
    query = input("Enter your research query: ")
    state = AgentState(query=query)
    final_state = app.invoke(state)
    if final_state.get("parsed_output"):
        print(final_state["parsed_output"])
    else:
        print("Could not parse output. Raw response:", final_state["llm_output"])