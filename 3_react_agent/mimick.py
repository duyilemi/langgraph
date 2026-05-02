from dotenv import load_dotenv
import datetime
import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ✅ New API imports (exactly as requested)
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.tools import tool

load_dotenv()

# -------------------------------
# LLM
# -------------------------------
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# -------------------------------
# Tools
# -------------------------------
@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current date and time in the specified format"""
    return datetime.datetime.now().strftime(format)

search_tool = TavilySearch()   # already a BaseTool

tools = [get_system_time, search_tool]

# Bind tools to the LLM so it can call them natively
llm_with_tools = llm.bind_tools(tools)

# -------------------------------
# State definition (conversation-based)
# -------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # chat history

# -------------------------------
# Nodes (Thinker and Doer)
# -------------------------------

def reason_node(state: AgentState):
    """
    Thinker: calls the LLM with the current conversation.
    The LLM will either respond with plain text (final answer)
    or request tool calls.
    """
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def act_node(state: AgentState):
    """
    Doer: executes the tool calls requested by the last LLM message
    and appends the results as ToolMessages.
    """
    messages = state["messages"]
    last_message = messages[-1]          # an AIMessage with optional tool_calls

    tool_results = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # Find the matching tool object
        tool_obj = None
        for t in tools:
            if t.name == tool_name:
                tool_obj = t
                break

        if tool_obj is not None:
            # Execute the tool (both dict and non-dict args work)
            result = tool_obj.invoke(tool_args)
        else:
            result = f"Tool '{tool_name}' not found"

        # Create a ToolMessage that pairs the result to the tool call id
        from langchain_core.messages import ToolMessage
        tool_results.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

    return {"messages": tool_results}


# -------------------------------
# Graph wiring (the ReAct loop)
# -------------------------------
def should_continue(state: AgentState):
    """After thinking, if the LLM wants to call tools → act, else stop."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "act_node"
    return END

graph = StateGraph(AgentState)

graph.add_node("reason_node", reason_node)
graph.add_node("act_node", act_node)

graph.set_entry_point("reason_node")
graph.add_conditional_edges("reason_node", should_continue)
graph.add_edge("act_node", "reason_node")   # always loop back after acting

app = graph.compile()

# -------------------------------
# Run the agent
# -------------------------------
response = app.invoke({
    "messages": [
        ("system", "You are a helpful assistant. Use tools when necessary."),
        ("user", "How many days ago was the latest SpaceX launch?")
    ]
})

# Final answer is the content of the last message (no tool calls)
print(response["messages"][-1].content, "final result")