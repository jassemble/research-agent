# Research Agent

> An autonomous AI research agent that searches the web, evaluates its own findings, and delivers a structured, human-readable report — powered by LangChain, LangGraph, and LangSmith.

---

## The Problem

When you ask an AI a research question, it usually does one of two things:

- **Hallucinate** — make up confident-sounding answers from stale training data
- **Dump raw text** — paste chunks of search results with no synthesis

Neither gives you a real answer. Research is iterative — you search, evaluate what you found, search again to fill gaps, then synthesise everything into something readable.

This agent does exactly that, autonomously.

---

## What It Does

You type a question. The agent:

1. Searches the web for relevant information
2. Evaluates whether it has enough to write a complete answer
3. Searches again if gaps remain (up to a configurable limit)
4. Breaks your question into its sub-questions
5. Finds specific answers to each one from the collected sources
6. Writes a clean, flowing report — not a bullet dump

The final report is printed to the terminal and saved as a timestamped Markdown file in `output/`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     LangGraph Graph                     │
│                                                         │
│   ┌──────────┐     ┌──────────┐     ┌───────────────┐  │
│   │  search  │────▶│ evaluate │────▶│  write_report │  │
│   │   node   │     │   node   │     │     node      │  │
│   └──────────┘     └────┬─────┘     └───────┬───────┘  │
│         ▲               │ not enough         │          │
│         └───────────────┘                   ▼          │
│                                            END          │
└─────────────────────────────────────────────────────────┘
         │                │                   │
         ▼                ▼                   ▼
    Tavily API         LLM call           2x LLM calls
    web search       JSON verdict        analyse + write
```

### Node Breakdown

| Node | Responsibility | Tools Used |
|---|---|---|
| `search` | Generates a focused search query from the question + prior coverage, fetches 5 results | LLM + Tavily |
| `evaluate` | Checks if every sub-question has a concrete answer; forces ≥ 2 search rounds | LLM |
| `write_report` | Two-pass synthesis: extract answers per sub-question → write flowing report | LLM × 2 |

### State Schema

```python
class AgentState(TypedDict):
    question: str              # the user's original question
    search_results: List[dict] # accumulated across all iterations
    iterations: int            # current search round count
    max_iterations: int        # hard stop (default: 3)
    enough_info: bool          # evaluate node's verdict
    report: str                # final written output
    messages: list             # full LLM conversation history
```

### Graph Routing

```
search → evaluate ──(enough = true)──▶ write_report → END
              │
         (enough = false,
          iterations < max)
              │
              ▼
           search   ← loops back
```

The conditional edge `route_after_evaluate` decides whether to loop or proceed. The evaluate node enforces a **minimum of 2 search rounds** before it can return `enough = true`.

---

## Report Generation: Two-Pass Approach

The `write_report` node uses two sequential LLM calls to avoid context overflow and force answer completeness:

```
Sources (capped at 500 chars each)
         │
         ▼
  ┌─────────────┐
  │  Call 1     │  "Break the question into sub-questions.
  │  Analyse    │   Answer each one from the sources."
  └──────┬──────┘
         │  bullet-point answers per sub-question
         ▼
  ┌─────────────┐
  │  Call 2     │  "Turn these answers into a clean,
  │  Write      │   flowing report with one section
  └──────┬──────┘   per sub-question."
         │
         ▼
    Final Report (Markdown)
```

This solves two problems:
- **Context overflow** — sources are trimmed; each call gets a manageable input
- **Unanswered sub-questions** — Call 1 is explicitly graded against every part of the question

---

## Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) | Stateful graph — nodes, edges, loops |
| LLM integration | [LangChain](https://github.com/langchain-ai/langchain) | Prompt templates, message types, LLM wrappers |
| LLM provider | Anthropic (`ChatAnthropic`) | Powers all reasoning calls |
| Web search | [Tavily](https://tavily.com) | Real-time web search API |
| Observability | [LangSmith](https://smith.langchain.com) | Full trace of every node, LLM call, and token |
| Output | Markdown files | Timestamped reports saved to `output/` |

---

## Project Structure

```
research-agent/
├── main.py               # Entry point — builds graph, runs it, saves report
├── requirements.txt
├── .env.example          # API key template
├── agent/
│   ├── state.py          # AgentState TypedDict
│   ├── tools.py          # Tavily search wrapper
│   ├── nodes.py          # search_node, evaluate_node, write_report_node
│   └── graph.py          # StateGraph wiring + conditional routing
└── output/               # Generated reports (auto-created)
    └── 2026-04-28_14-32_your-question-here.md
```

---

## Setup

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your keys:

```env
# LLM
ANTHROPIC_BASE_URL=https://your-llm-proxy.com
ANTHROPIC_AUTH_TOKEN=your_token_here

# Web search — free tier at https://app.tavily.com
TAVILY_API_KEY=your_tavily_key_here

# Observability — free tier at https://smith.langchain.com
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=research-agent
```

### 3. Run

```bash
python3 main.py
```

```
============================================================
  Research Agent
  Powered by LangChain + LangGraph + LangSmith
============================================================

Enter your research question:
> Who is Andrej Karpathy and what is his LLM Wiki?

Starting research (max 3 search rounds)...

[Search #1] Query: Andrej Karpathy biography AI researcher
[Search #1] Found 5 results
[Evaluate] Enough info? No — wiki content not yet covered
[Search #2] Query: Andrej Karpathy LLM Wiki concept how it works
[Search #2] Found 5 results
[Evaluate] Enough info? Yes

[Write Report] Extracting answers to each question...
[Write Report] Writing final report...

============================================================
RESEARCH REPORT
============================================================
...

Report saved to: output/2026-04-28_14-32_who-is-andrej-karpathy.md
============================================================
```

---

## Observability with LangSmith

When `LANGCHAIN_TRACING_V2=true`, every run is automatically traced. You get:

- Full node execution timeline
- Every prompt sent and response received
- Token counts and latency per call
- The complete agent loop visualised as a tree

View traces at [smith.langchain.com](https://smith.langchain.com) under your project name.

---

## Configuration

| Variable | Location | Default | Description |
|---|---|---|---|
| `max_iterations` | `main.py` line 25 | `3` | Max search rounds before forcing report |
| Source content cap | `nodes.py` | `500 chars` | Characters kept per source in report node |
| Search results per round | `nodes.py` | `5` | Results fetched from Tavily per search |
| LLM max tokens | `nodes.py` | `4096` | Max tokens in LLM responses |
