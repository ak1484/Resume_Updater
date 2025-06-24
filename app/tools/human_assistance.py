from langchain_core.tools import tool

@tool("human_assistance_tool")
def human_assistance_tool(query: str) -> str:
    """Request assistance from a human."""
    return f"Human assistance requested for: {query}" 