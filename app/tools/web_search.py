from langchain_tavily import TavilySearch
from langchain_core.tools import tool

ALLOWED_TOPICS = {"general", "news", "finance","sports"}

@tool
def web_search(query: str, topic: str = "general", **kwargs):
    """A tool for searching the web for up-to-date information. Allowed topics: 'general', 'news', 'finance'."""
    if topic not in ALLOWED_TOPICS:
        topic = "general"
    return TavilySearch(
        max_results=2,
        name="web_search",
        description="A tool for searching the web for up-to-date information. Allowed topics: 'general', 'news', 'finance'. Use this tool to answer questions about current events, news, or anything not in your training data."
    ).invoke({"query": query, "topic": topic, **kwargs}) 