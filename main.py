import os
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# LangSmith tracing is enabled automatically via env vars —
# LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT

from agent.graph import build_graph


def main():
    graph = build_graph()

    print("=" * 60)
    print("  Research Agent")
    print("  Powered by LangChain + LangGraph + LangSmith")
    print("=" * 60)

    question = input("\nEnter your research question:\n> ").strip()
    if not question:
        print("No question provided. Exiting.")
        return

    max_iterations = 3  # change to dig deeper

    print(f"\nStarting research (max {max_iterations} search rounds)...\n")

    initial_state = {
        "question": question,
        "search_results": [],
        "iterations": 0,
        "max_iterations": max_iterations,
        "enough_info": False,
        "report": "",
        "messages": [],
    }

    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("RESEARCH REPORT")
    print("=" * 60)
    print(final_state["report"])

    # Save report to output/
    slug = re.sub(r"[^\w\s-]", "", question.lower())
    slug = re.sub(r"\s+", "-", slug)[:60]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{timestamp}_{slug}.md"
    output_path.write_text(
        f"# {question}\n\n{final_state['report']}\n\n"
        f"---\n*Sources consulted: {len(final_state['search_results'])} | "
        f"Search rounds: {final_state['iterations']}*\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print(f"Completed in {final_state['iterations']} search round(s)")
    print(f"Sources consulted: {len(final_state['search_results'])}")
    print(f"Report saved to: {output_path}")
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        project = os.getenv("LANGCHAIN_PROJECT", "research-agent")
        print(f"Trace available in LangSmith project: '{project}'")
    print("=" * 60)


if __name__ == "__main__":
    main()
