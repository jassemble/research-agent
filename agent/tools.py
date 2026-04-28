import os
from tavily import TavilyClient


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web using Tavily and return a list of results."""
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = client.search(query, max_results=max_results, include_raw_content=False)
    results = response.get("results", [])
    # Normalize to only the fields we need
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        }
        for r in results
    ]
