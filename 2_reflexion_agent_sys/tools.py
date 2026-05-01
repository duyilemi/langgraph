from langchain_tavily import TavilySearch

tavily = TavilySearch(max_results=2)


def run_search(queries: list[str]) -> dict:
    results = {}
    for q in queries:
        results[q] = tavily.invoke(q)
    return results