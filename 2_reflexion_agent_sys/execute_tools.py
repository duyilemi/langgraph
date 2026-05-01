import json
from typing import List
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_tavily import TavilySearch

tavily = TavilySearch(max_results=2)


def execute_tools(state: dict) -> dict:
    messages: List[BaseMessage] = state["messages"]
    last_message = messages[-1]

    if not isinstance(last_message, AIMessage):
        return {"messages": []}

    tool_calls = last_message.additional_kwargs.get("tool_calls", [])

    tool_messages = []

    for tool_call in tool_calls:
        args = tool_call.get("args", {})
        queries = args.get("search_queries", [])

        results = {}
        for q in queries:
            results[q] = tavily.invoke(q)

        tool_messages.append(
            ToolMessage(
                content=json.dumps(results),
                tool_call_id=tool_call["id"]
            )
        )

    return {"messages": tool_messages}