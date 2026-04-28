import json
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from .state import AgentState
from .tools import search_web


def _text(content) -> str:
    """Extract text from LLM response — handles plain strings, text blocks, and thinking blocks."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if text_parts:
            return "".join(text_parts).strip()
        # Model returned only thinking blocks — extract that content
        thinking_parts = [
            block.get("thinking", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "thinking"
        ]
        return "".join(thinking_parts).strip()
    return str(content).strip()


def _is_thinking_only(content) -> bool:
    """Return True if the response has thinking blocks but no text blocks."""
    if not isinstance(content, list):
        return False
    has_text = any(
        isinstance(b, dict) and b.get("type") == "text" and b.get("text", "").strip()
        for b in content
    )
    has_thinking = any(
        isinstance(b, dict) and b.get("type") == "thinking"
        for b in content
    )
    return has_thinking and not has_text


llm = ChatAnthropic(
    model="geekyants/premium",
    anthropic_api_key=os.getenv("ANTHROPIC_AUTH_TOKEN"),
    anthropic_api_url=os.getenv("ANTHROPIC_BASE_URL"),
    temperature=0,
    max_tokens=4096,
)


# ---------------------------------------------------------------------------
# Node 1 — Search
# ---------------------------------------------------------------------------

def search_node(state: AgentState) -> dict:
    """Generate a search query and fetch results from the web."""
    question = state["question"]
    previous_results = state["search_results"]
    iteration = state["iterations"] + 1

    already_covered = (
        "\n".join(f"- {r['title']}" for r in previous_results)
        if previous_results
        else "None yet"
    )

    messages = [
        SystemMessage(content=(
            "You are a research assistant. Your job is to generate a precise web search query "
            "to gather information on the user's question. "
            "Return ONLY the search query string — no explanation, no quotes."
        )),
        HumanMessage(content=(
            f"Research question: {question}\n\n"
            f"Topics already covered (iteration {iteration}):\n{already_covered}\n\n"
            "Generate a search query that fills gaps or digs deeper."
        )),
    ]

    response = llm.invoke(messages)
    query = _text(response.content)
    print(f"\n[Search #{iteration}] Query: {query}")

    results = search_web(query, max_results=5)
    print(f"[Search #{iteration}] Found {len(results)} results")

    return {
        "search_results": previous_results + results,
        "iterations": iteration,
        "messages": messages + [response],
    }


# ---------------------------------------------------------------------------
# Node 2 — Evaluate
# ---------------------------------------------------------------------------

def evaluate_node(state: AgentState) -> dict:
    """Decide whether we have enough information to write a report."""
    question = state["question"]
    results = state["search_results"]
    iterations = state["iterations"]
    max_iterations = state["max_iterations"]

    if iterations >= max_iterations:
        print(f"\n[Evaluate] Reached max iterations ({max_iterations}). Writing report.")
        return {"enough_info": True}

    # Require at least 2 iterations before allowing early exit
    if iterations < 2:
        print(f"[Evaluate] Only {iterations} round(s) done — forcing another search.")
        return {"enough_info": False}

    sources_summary = "\n\n".join(
        f"Source {i+1}: {r['title']}\n{r['content'][:300]}"
        for i, r in enumerate(results)
    )

    messages = [
        SystemMessage(content=(
            "You are a research quality evaluator. "
            "The user asked a multi-part question. Check whether EVERY part of the question "
            "has a clear, specific answer in the sources. "
            "Only return true if ALL sub-questions are answered with concrete details — "
            "not vague summaries. "
            'Reply with JSON only: {"enough": true} or {"enough": false, "reason": "..."}'
        )),
        HumanMessage(content=(
            f"Question: {question}\n\n"
            f"Sources ({len(results)} total):\n{sources_summary}"
        )),
    ]

    response = llm.invoke(messages)
    raw = _text(response.content)

    try:
        clean = raw.strip("`").strip()
        if clean.startswith("json"):
            clean = clean[4:].strip()
        parsed = json.loads(clean)
        enough = parsed.get("enough", False)
        reason = parsed.get("reason", "")
    except json.JSONDecodeError:
        enough = "true" in raw.lower()
        reason = ""

    status = "Yes" if enough else f"No — {reason}"
    print(f"[Evaluate] Enough info? {status}")

    return {"enough_info": enough, "messages": [response]}


# ---------------------------------------------------------------------------
# Node 3 — Write Report
# ---------------------------------------------------------------------------

def write_report_node(state: AgentState) -> dict:
    """Synthesise all search results into a structured research report."""
    question = state["question"]
    results = state["search_results"]

    sources_text = "\n\n".join(
        f"[{i+1}] {r['title']}\n{r['content'][:500]}"
        for i, r in enumerate(results)
    )

    # --- Call 1: extract answers to each sub-question in the user's query ---
    print("\n[Write Report] Extracting answers to each question...")
    analyse_messages = [
        SystemMessage(content=(
            "You are a research analyst. The user asked a question that may contain "
            "multiple sub-questions. Your job:\n"
            "1. Break the user's question into its individual sub-questions.\n"
            "2. For each sub-question, find the best answer from the sources.\n"
            "3. Quote or paraphrase the most relevant source content directly.\n"
            "Be specific and complete. Do not skip any sub-question."
        )),
        HumanMessage(content=(
            f"User question: {question}\n\n"
            f"Sources:\n{sources_text}\n\n"
            "Answer each sub-question from the sources."
        )),
    ]
    analysis_response = llm.invoke(analyse_messages)
    analysis = _text(analysis_response.content)

    # --- Call 2: write a clean, flowing report from the answers ---
    print("[Write Report] Writing final report...")
    write_messages = [
        SystemMessage(content=(
            "You are a research writer. You have been given a set of answers to sub-questions. "
            "Your job is to turn them into a clean, well-written report that a non-expert "
            "can read and immediately understand.\n\n"
            "Rules:\n"
            "- Use a clear heading for each sub-question's answer\n"
            "- Write in full paragraphs — no bullet dumps\n"
            "- Be specific: include names, details, examples from the research\n"
            "- Do NOT truncate or summarize vaguely — fully answer every question\n"
            "- Output ONLY the final report text, nothing else"
        )),
        HumanMessage(content=(
            f"Original question: {question}\n\n"
            f"Research answers:\n{analysis}\n\n"
            "Write the final report now."
        )),
    ]
    report_response = llm.invoke(write_messages)
    report = _text(report_response.content)

    if _is_thinking_only(report_response.content):
        report = analysis

    return {"report": report, "messages": [analysis_response, report_response]}
