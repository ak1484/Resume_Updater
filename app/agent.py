import os
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel
from dotenv import load_dotenv
import json
import re
from langchain_core.messages import ToolMessage

# Load environment variables from .env
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# LangChain Gemini LLM
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    # If not installed, user must run: pip install langchain-google-genai
    ChatGoogleGenerativeAI = None

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langsmith import traceable
from langchain_core.messages import AIMessage, HumanMessage

# LangSmith (for tracing)
import langsmith

# Set up Gemini via LangChain
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY and ChatGoogleGenerativeAI:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)
else:
    llm = None

# Set up Google Custom Search tool
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
if GOOGLE_API_KEY and GOOGLE_CSE_ID:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_CSE_ID"] = GOOGLE_CSE_ID

# Set up Tavily API key from .env
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Create TavilySearch tool with explicit name and description for Gemini tool-calling
# This helps the LLM recognize it as a web search tool
# DOC: This tool allows the agent to search the web for up-to-date information when its own knowledge is insufficient.
tavily_tool = TavilySearch(
    max_results=2,
    name="web_search",
    description="A tool for searching the web for up-to-date information. Use this tool to answer questions about current events, news, or anything not in your training data."
)

# Human-in-the-loop tool
@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    # In a real app, this would pause and wait for human input
    return f"Human assistance requested for: {query}"

# State definition
class State(TypedDict):
    messages: Annotated[list, add_messages]
    # Add more custom state fields as needed

class SupervisorDecision(BaseModel):
    action: str  # 'chatbot', 'tools', or 'end'
    reason: str

# Build the graph
memory = MemorySaver()
graph_builder = StateGraph(State)

tools = [tavily_tool, human_assistance]
if llm:
    llm_with_tools = llm.bind_tools(tools)
else:
    llm_with_tools = None

def debug_tool_node(inputs):
    # This wraps the ToolNode to print when a tool is called
    tool_node = ToolNode(tools=tools)
    if "messages" in inputs and hasattr(inputs["messages"][-1], "tool_calls"):
        tool_calls = inputs["messages"][-1].tool_calls
        if tool_calls:
            print(f"[DEBUG] Tool(s) invoked: {[call['name'] for call in tool_calls]}")
    return tool_node.invoke(inputs)

def chatbot(state: State):
    if llm_with_tools:
        message = llm_with_tools.invoke(state["messages"])
        return {"messages": [message]}
    else:
        return {"messages": [AIMessage(content="Gemini API key not set or model unavailable.")]}

def supervisor_router(state: State):
    # Use Gemini to classify the user query and decide routing
    from langchain_core.messages import HumanMessage
    if not state["messages"]:
        return END
    last_msg = state["messages"][-1]
    if isinstance(last_msg, HumanMessage):
        # List available tools and their descriptions
        tool_descriptions = (
            "Available tools:\n"
            "- web_search: A tool for searching the web for up-to-date information. Use this tool to answer questions about current events, news, or anything not in your training data.\n"
            "- human_assistance: A tool to request help from a human when the agent is unsure or needs human input.\n"
        )
        if llm:
            prompt = (
                f"{tool_descriptions}"
                "You are a supervisor for an agent with access to these tools. "
                "If the user query is about current events, news, sports, or anything that could be checked on the web, "
                "ALWAYS choose action: 'tools' (even if you think the answer is not available or is in the future). "
                "Only choose 'chatbot' for general conversation or questions that do not require up-to-date information. "
                "Return a JSON object with an 'action' key (one of: 'chatbot', 'tools', 'end') and a 'reason' key. "
                f"User query: {last_msg.content}"
            )
            # Use Gemini's structured output
            try:
                response = llm.invoke(prompt)
                content = response.content
                print(f"[SUPERVISOR] Raw response: {content}")
                if isinstance(content, list):
                    content = " ".join(str(x) for x in content)
                # Try to extract JSON from code block or text
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = content  # fallback
                data = json.loads(json_str)
                decision = SupervisorDecision.parse_obj(data)
                action = decision.action
                print(f"[SUPERVISOR] Decision: {action}, Reason: {decision.reason}")
                if action == "chatbot":
                    return "chatbot"
                elif action == "tools":
                    return "invoke_tool"
                else:
                    return END
            except Exception as e:
                print(f"[SUPERVISOR] Error in structured output: {e}")
                return "chatbot"
        else:
            return "chatbot"
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END

def invoke_tool_node(state: State):
    if llm_with_tools:
        # Add a system message to encourage tool use
        system_message = HumanMessage(content="Use the web_search tool for any query about news, current events, or sports, even if you think the answer is not available.")
        messages = [system_message] + state["messages"]
        # Filter out empty messages
        messages = [m for m in messages if getattr(m, 'content', None) and isinstance(m.content, str) and m.content.strip()]
        tool_message = llm_with_tools.invoke(messages)
        print(f"[INVOKE_TOOL_NODE] tool_calls: {getattr(tool_message, 'tool_calls', None)}")
        updated_messages = state["messages"] + [tool_message]
        from langchain_core.messages import AIMessage
        summary = None
        if isinstance(tool_message, AIMessage) and getattr(tool_message, "tool_calls", None):
            for tool_call in tool_message.tool_calls:
                if tool_call["name"] == "web_search":
                    tool_result = tavily_tool.invoke(tool_call["args"])
                    # Summarize the results with the LLM
                    articles = tool_result.get("results", []) 
                    if articles:
                        # Compose a summary prompt
                        articles_text = "\n\n".join([
                            f"Title: {a.get('title', '')}\nContent: {a.get('content', '')}" for a in articles
                        ])
                        summary_prompt = (
                            "Summarize the following news articles into a single, concise news update for the user. "
                            "Focus on the most important facts and context.\n\n" + articles_text
                        )
                        if llm:
                            # Guard against empty summary prompt
                            if isinstance(summary_prompt, str) and summary_prompt.strip():
                                summary_msg = llm.invoke([HumanMessage(content=summary_prompt)])
                                summary = summary_msg.content if hasattr(summary_msg, 'content') else str(summary_msg)
                                updated_messages.append(AIMessage(content=summary))
                            else:
                                updated_messages.append(AIMessage(content="[ERROR] No content to summarize."))
                        else:
                            updated_messages.append(AIMessage(content="[ERROR] Summarization unavailable: Gemini API key not set or model unavailable."))
                    else:
                        updated_messages.append(ToolMessage(
                            content=json.dumps(tool_result),
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"]
                        ))
                elif tool_call["name"] == "human_assistance":
                    tool_result = human_assistance.invoke(tool_call["args"])
                    updated_messages.append(
                        ToolMessage(
                            content=json.dumps(tool_result),
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"]
                        )
                    )
                else:
                    tool_result = f"Unknown tool: {tool_call['name']}"
                    updated_messages.append(
                        ToolMessage(
                            content=json.dumps(tool_result),
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"]
                        )
                    )
        return {"messages": updated_messages}
    else:
        return {"messages": state["messages"]}

def supervisor_node(state: State):
    # This node just returns the state unchanged
    return state

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("supervisor", supervisor_node)
graph_builder.add_node("invoke_tool", invoke_tool_node)

graph_builder.add_edge(START, "supervisor")
graph_builder.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {"tools": "invoke_tool", "chatbot": "chatbot", "invoke_tool": "invoke_tool", END: END}
)
graph_builder.add_edge("invoke_tool", "supervisor")

graph = graph_builder.compile(checkpointer=memory)

# LangSmith tracing (optional, requires setup)
langsmith_client = langsmith.Client()

# Agent interface for FastAPI
@traceable(name="agent_invoke")
def run_agent(query: str, thread_id: str = "default") -> str:
    # Guard against empty queries
    if not query or not query.strip():
        return "Please enter a non-empty query."
    # Per LangGraph docs and official examples, config should be {'configurable': {'thread_id': ...}}
    config = {"configurable": {"thread_id": thread_id}}
    # Filter out empty messages before passing to the graph
    initial_messages = [HumanMessage(content=query)]
    initial_messages = [m for m in initial_messages if getattr(m, 'content', None) and isinstance(m.content, str) and m.content.strip()]
    events = graph.stream({"messages": initial_messages}, config, stream_mode="values")
    result = ""
    for event in events:
        if "messages" in event:
            result = event["messages"][-1].content
    return result 