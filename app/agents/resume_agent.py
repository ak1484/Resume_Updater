from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel
import json
import re
from langchain_core.messages import ToolMessage

from app.utils.env import GEMINI_API_KEY
from langsmith import traceable

import os
# print("LANGCHAIN_API_KEY at runtime:", os.getenv("LANGCHAIN_API_KEY"))

from app.tools.web_search import web_search
from app.tools.human_assistance import human_assistance_tool
from app.memory.memory import memory_saver

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage

from langsmith import Client
langsmith_client = Client()

if GEMINI_API_KEY and ChatGoogleGenerativeAI:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)
else:
    llm = None

tools = [web_search, human_assistance_tool]
if llm:
    llm_with_tools = llm.bind_tools(tools)
else:
    llm_with_tools = None

class State(TypedDict):
    messages: Annotated[list, add_messages]

class SupervisorDecision(BaseModel):
    action: str
    reason: str

def chatbot(state: State):
    if llm_with_tools:
        filtered_messages = [m for m in state["messages"] if getattr(m, 'content', None) and isinstance(m.content, str) and m.content.strip()]
        message = llm_with_tools.invoke(filtered_messages)
        return {"messages": [message]}
    else:
        return {"messages": [AIMessage(content="Gemini API key not set or model unavailable.")]}

def supervisor_router(state: State):
    from langchain_core.messages import HumanMessage
    if not state["messages"]:
        return END
    last_msg = state["messages"][-1]
    if isinstance(last_msg, HumanMessage):
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
            try:
                response = llm.invoke(prompt)
                content = response.content
                print(f"[SUPERVISOR] Raw response: {content}")
                if isinstance(content, list):
                    content = " ".join(str(x) for x in content)
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = content
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
        system_message = HumanMessage(content="Use the web_search tool for any query about news, current events, or sports, even if you think the answer is not available.")
        messages = [system_message] + state["messages"]
        messages = [m for m in messages if getattr(m, 'content', None) and isinstance(m.content, str) and m.content.strip()]
        tool_message = llm_with_tools.invoke(messages)
        print(f"[INVOKE_TOOL_NODE] tool_calls: {getattr(tool_message, 'tool_calls', None)}")
        updated_messages = state["messages"] + [tool_message]
        from langchain_core.messages import AIMessage
        summary = None
        if isinstance(tool_message, AIMessage) and getattr(tool_message, "tool_calls", None):
            for tool_call in tool_message.tool_calls:
                if tool_call["name"] == "web_search":
                    tool_result = web_search.invoke(tool_call["args"])
                    articles = tool_result.get("results", []) 
                    if articles:
                        articles_text = "\n\n".join([
                            f"Title: {a.get('title', '')}\nContent: {a.get('content', '')}" for a in articles
                        ])
                        summary_prompt = (
                            "Summarize the following news articles into a single, concise news update for the user. "
                            "Focus on the most important facts and context.\n\n" + articles_text
                        )
                        if llm:
                            if isinstance(summary_prompt, str) and summary_prompt.strip():
                                summary_msg = llm.invoke([HumanMessage(content=summary_prompt)])
                                summary = summary_msg.content if hasattr(summary_msg, 'content') else str(summary_msg)
                                updated_messages.append(AIMessage(content=summary))
                                return {"messages": updated_messages + [AIMessage(content=summary)]}
                            else:
                                updated_messages.append(AIMessage(content="[ERROR] No content to summarize."))
                                return {"messages": updated_messages}
                        else:
                            updated_messages.append(AIMessage(content="[ERROR] Summarization unavailable: Gemini API key not set or model unavailable."))
                            return {"messages": updated_messages}
                    else:
                        updated_messages.append(AIMessage(content="No articles found."))
                        return {"messages": updated_messages}
                elif tool_call["name"] == "human_assistance_tool":
                    tool_result = human_assistance_tool.invoke(tool_call["args"])
                    updated_messages.append(AIMessage(content=tool_result))
                    return {"messages": updated_messages}
                else:
                    tool_result = f"Unknown tool: {tool_call['name']}"
                    updated_messages.append(AIMessage(content=tool_result))
                    return {"messages": updated_messages}
        return {"messages": updated_messages}
    else:
        return {"messages": state["messages"]}

def supervisor_node(state: State):
    return state

graph_builder = StateGraph(State)
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
graph = graph_builder.compile(checkpointer=memory_saver)

@traceable(name="resume_agent_invoke")
def run_agent(query: str, thread_id: str = "1") -> str:
    if not query or not query.strip():
        return "Please enter a non-empty query."
    config = {"configurable": {"thread_id": thread_id}}
    initial_messages = [HumanMessage(content=query)]
    initial_messages = [m for m in initial_messages if getattr(m, 'content', None) and isinstance(m.content, str) and m.content.strip()]
    # print("Initial messages:", initial_messages)
    events = graph.stream({"messages": initial_messages}, config, stream_mode="values")  # type: ignore
    result = ""
    for event in events:
        print("Graph event:", event)
        if "messages" in event:
            result = event["messages"][-1].content
    if not result or not isinstance(result, str) or not result.strip():
        return "No response generated."
    return result 