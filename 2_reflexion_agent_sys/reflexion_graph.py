from typing import TypedDict, List

from langgraph.graph import StateGraph, END

from langchain_core.messages import BaseMessage, HumanMessage

from chains import draft_chain, revise_chain
from tools import run_search


# =========================
# STATE
# =========================
class State(TypedDict):
    messages: List[BaseMessage]
    draft: dict
    search_results: dict
    revision: dict
    iteration: int


MAX_ITER = 1


# =========================
# NODES
# =========================

def draft_node(state: State):
    result = draft_chain.invoke({
        "messages": state["messages"]
    })

    return {
        "draft": result,
        "iteration": 0
    }


def search_node(state: State):
    queries = state["draft"].search_queries
    results = run_search(queries)

    return {
        "search_results": results
    }


def revise_node(state: State):
    result = revise_chain.invoke({
        "messages": state["messages"] + [
            HumanMessage(content=f"""
Previous Answer:
{state['draft'].answer}

Critique:
Missing: {state['draft'].reflection.missing}
Superfluous: {state['draft'].reflection.superfluous}

Search Results:
{state['search_results']}
""")
        ]
    })

    return {
        "revision": result,
        "iteration": state["iteration"] + 1
    }


# =========================
# CONTROL FLOW
# =========================
def should_continue(state: State):
    if state["iteration"] >= MAX_ITER:
        return END
    return "search"


# =========================
# GRAPH
# =========================
graph = StateGraph(State)

graph.add_node("draft", draft_node)
graph.add_node("search", search_node)
graph.add_node("revise", revise_node)

graph.set_entry_point("draft")

graph.add_edge("draft", "search")
graph.add_edge("search", "revise")

graph.add_conditional_edges("revise", should_continue)

app = graph.compile()


# =========================
# RUN
# =========================
if __name__ == "__main__":
    result = app.invoke({
        "messages": [
            HumanMessage(
                content="How can small businesses use AI to grow?"
            )
        ]
    })

    print("\nFINAL ANSWER:\n")
    print(result["revision"].answer)

    print("\nREFERENCES:\n")
    print(result["revision"].references)