from dotenv import load_dotenv
import datetime
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()

# -------------------------------
# LLM
# -------------------------------
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# -------------------------------
# Tools (same as before, just with updated import)
# -------------------------------
@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current date and time in the specified format"""
    return datetime.datetime.now().strftime(format)

search_tool = TavilySearch()

tools = [get_system_time, search_tool]

# -------------------------------
# Agent – replaces the entire StateGraph
# -------------------------------
agent = create_agent(
    model=llm,
    tools=tools
)

# -------------------------------
# Run the agent
# -------------------------------
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "How many days ago was the latest SpaceX launch?"}
    ]
})

print(response["messages"][-1].content, "final result")