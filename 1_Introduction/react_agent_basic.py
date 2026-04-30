from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv
import datetime

load_dotenv()

# ✅ Updated model
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Tools
search_tool = TavilySearch()

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current date and time in the specified format"""
    return datetime.datetime.now().strftime(format)

tools = [search_tool, get_system_time]

# ✅ New agent API
agent = create_agent(
    model=llm,
    tools=tools
)

# Run
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "When was SpaceX's last launch and how many days ago was that from this instant"}
    ]
})

print(response["messages"][-1].content)