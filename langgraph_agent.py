"""LangGraph tool-calling agent with conditional routing and thread-scoped memory.

Flow:  START -> llm -> (tools -> llm)* -> END"""

from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers a and b and return the result."""
    return a * b


@tool
def word_length(word: str) -> int:
    """Return the number of characters in a word."""
    return len(word)


tools = [multiply, word_length]

llm = init_chat_model("groq:openai/gpt-oss-120b")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


builder = StateGraph(State)
builder.add_node("llm", chatbot)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition, {"tools": "tools", END: END})
builder.add_edge("tools", "llm")
graph = builder.compile(checkpointer=MemorySaver())


def ask(question: str, thread_id: str = "demo"):
    """Send one question and print each step of the loop."""
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\nUSER: {question}")
    for event in graph.stream(
        {"messages": [{"role": "user", "content": question}]},
        config,
        stream_mode="values",
    ):
        msg = event["messages"][-1]
        kind = getattr(msg, "type", "?")
        if kind == "ai" and getattr(msg, "tool_calls", None):
            calls = ", ".join(f"{c['name']}({c['args']})" for c in msg.tool_calls)
            print(f"  [llm]  -> wants tool: {calls}")
        elif kind == "tool":
            print(f"  [tool] -> {msg.content}")
        elif kind == "ai":
            print(f"  [llm]  -> FINAL: {msg.content}")


if __name__ == "__main__":
    ask("What is 23 times 19, and how many letters are in the word 'agentic'?")
    ask("Multiply that first result by 2.", thread_id="demo")
