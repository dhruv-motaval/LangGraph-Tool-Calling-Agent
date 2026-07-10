# LangGraph Tool-Calling Agent

A minimal, readable tool-calling agent built from **LangGraph primitives**
(`StateGraph`, `ToolNode`, `tools_condition`) instead of prebuilt wrappers — so the
agentic loop is explicit and easy to understand.

The model decides whether it needs a tool, the tool runs, and control loops back to
the model until it has enough information to answer:

```
START ──▶ LLM ──▶ (tools ──▶ LLM)* ──▶ END
```

That loop back from the tool node to the LLM is the single thing that turns a
chatbot into an agent.

---

## Features

- **Tool-calling loop** with conditional routing (`llm → tools → llm`)
- **Multi-step tool use** — chains several tool calls before producing a final answer
- **Per-conversation memory** via a `MemorySaver` checkpointer (thread-scoped state), so follow-up questions reuse earlier results
- **Streamed steps** — prints each hop of the loop so you can watch it reason

## Tech stack

`Python` · `LangGraph` · `LangChain` · `Groq`

---

## Getting started

### Prerequisites
- Python 3.10+
- A Groq API key ([console.groq.com](https://console.groq.com))

### Install

```bash
pip install -r requirements.txt
```

### Configure

Copy the example env file and add your key:

```bash
cp .env.example .env
```

```
GROQ_API_KEY=your_key_here
```

### Run

```bash
python langgraph_agent.py
```

---

## Example

Input:

```
What is 23 times 19, and how many letters are in the word "agentic"?
```

Streamed trace:

```
[llm]  -> wants tool: multiply({'a': 23, 'b': 19})
[tool] -> 437
[llm]  -> wants tool: word_length({'word': 'agentic'})
[tool] -> 7
[llm]  -> FINAL: 23 times 19 is 437, and "agentic" has 7 letters.
```

The agent made two separate tool calls and combined the results — that's the
multi-step loop in action.

---

## How it works

1. **State** — `State` holds the running message history. The `add_messages`
   reducer appends new messages instead of overwriting them.
2. **LLM node** — calls the model (bound to the tools). The model returns either a
   normal answer or a message containing tool calls.
3. **Routing** — `tools_condition` inspects the model's last message: if it contains
   tool calls it routes to the tool node, otherwise it routes to `END`.
4. **Tool node** — `ToolNode` executes whatever tool the model asked for and adds the
   result to the state.
5. **The loop** — an edge from `tools` back to `llm` sends the result to the model so
   it can continue. Remove that one edge and you have a plain chatbot.
6. **Memory** — compiling with a `MemorySaver` checkpointer gives each conversation
   (`thread_id`) its own persistent state across turns.

## Project structure

```
.
├── langgraph_agent.py    # the agent: state, tools, graph, and a demo runner
├── requirements.txt      # dependencies
├── .env.example          # template for your GROQ_API_KEY
├── .env                  # your actual key (git-ignored, not committed)
└── README.md
```

## What this demonstrates

- Building an agent from LangGraph primitives, not a black-box helper
- Conditional routing and the tool-execution loop
- Stateful, multi-turn conversations with checkpointed memory

## Next steps

- Add real tools (web search, a calculator, an API client)
- Extend to a multi-agent setup: a supervisor node that routes between specialist agents
