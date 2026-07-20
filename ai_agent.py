"""
Custom Research AI Agent (OpenRouter version)
================================================
Uses OpenRouter's free gpt-oss-20b model instead of Claude.
Same agent loop pattern: model decides to search -> we run the search ->
feed results back -> model repeats until it writes a final report.

Requirements:
    pip install requests ddgs --break-system-packages

Setup (IMPORTANT - never hardcode your key in the script):
    export OPENROUTER_API_KEY="your-key-here"

Usage:
    python ai_agent.py "impact of AI on education"
"""

import os
import sys
import json
import requests
from ddgs import DDGS
from dotenv import load_dotenv

load_dotenv()  # reads the .env file and loads variables into the environment

API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = "openai/gpt-oss-20b:free"
URL = "https://openrouter.ai/api/v1/chat/completions"


# ---------------------------------------------------------------------
# TOOL: Web Search
# ---------------------------------------------------------------------
def web_search(query: str, max_results: int = 5):
    """Search the web and return a list of {title, url, snippet} dicts."""
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    {
                        "title": r.get("title"),
                        "url": r.get("href"),
                        "snippet": r.get("body"),
                    }
                )
    except Exception as e:
        results.append({"error": str(e)})
    return results


# OpenAI-style tool/function definition (OpenRouter follows this schema)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for information on a topic. Returns titles, "
                "URLs, and short snippets. Call this multiple times with "
                "different queries to research a topic from different angles."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
        },
    }
]


def call_model(messages):
    """Send a chat completion request to OpenRouter."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "Research Agent",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "max_tokens": 2000,
    }
    response = requests.post(URL, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"Request failed ({response.status_code}): {response.text}")
    return response.json()


# ---------------------------------------------------------------------
# AGENT LOOP
# ---------------------------------------------------------------------
def run_agent(topic: str, max_turns: int = 8, on_search=None):
    """
    on_search: optional callback(query: str) called every time the agent
    performs a search — used by the frontend to show live progress.
    """
    messages = [
        {
            "role": "user",
            "content": (
                f"Research the topic: '{topic}'.\n\n"
                "Use the web_search tool as many times as needed to gather "
                "information from at least 2-3 different angles or sources. "
                "Once you have enough information, STOP searching and write a "
                "well-structured final report with these sections:\n"
                "1. Summary (2-3 sentences)\n"
                "2. Key Findings (bullet points)\n"
                "3. Different Perspectives / Debates (if any)\n"
                "4. Sources Used (list the URLs you found)\n"
            ),
        }
    ]

    for turn in range(max_turns):
        data = call_model(messages)

        try:
            choice = data["choices"][0]["message"]
        except (KeyError, IndexError):
            return f"Unexpected response format:\n{data}"

        messages.append(choice)

        tool_calls = choice.get("tool_calls")

        # No tool calls -> model is giving its final answer
        if not tool_calls:
            content = choice.get("content")
            if content and content.strip():
                return content.strip()

            # The free model sometimes stops without writing the report.
            # Give it one explicit nudge to produce the final text now.
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You have not written the report yet. Based on all the "
                        "search results above, write the final report now in "
                        "plain text with the sections requested earlier. "
                        "Do not call any more tools."
                    ),
                }
            )
            retry_data = call_model(messages)
            try:
                retry_choice = retry_data["choices"][0]["message"]
                retry_content = retry_choice.get("content")
                if retry_content and retry_content.strip():
                    return retry_content.strip()
            except (KeyError, IndexError):
                pass

            return "(model returned empty content twice — try increasing max_tokens or rerunning)"

        # Otherwise, execute each requested tool call
        for tc in tool_calls:
            fn_name = tc["function"]["name"]

            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                # The free model occasionally returns malformed arguments.
                # Don't crash the whole job — just report it back to the model.
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(
                            {"error": "Malformed arguments, please retry the search with a clean query."}
                        ),
                    }
                )
                continue

            if fn_name == "web_search":
                query = args.get("query", "")
                print(f"🔍 Searching: {query}")
                if on_search:
                    on_search(query)
                results = web_search(query)
            else:
                results = {"error": f"Unknown tool {fn_name}"}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(results),
                }
            )

    return "⚠️ Max turns reached without a final report. Try increasing max_turns."


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    if not API_KEY:
        raise SystemExit(
            "Missing OPENROUTER_API_KEY. Set it is environment variable "
            "before running this script:\n"
            "  export OPENROUTER_API_KEY='your-key-here'\n"
            "Get a free key at https://openrouter.ai/keys"
        )

    if len(sys.argv) < 2:
        topic = "Explain latest advancements in AI and their potential impact on society."
        print(f"No topic given, using default:\n{topic}\n")
    else:
        topic = " ".join(sys.argv[1:])

    print(f"🧠 Starting research on: {topic}\n")

    report = run_agent(topic)

    print("\n📄 FINAL REPORT:\n")
    print(report)

    filename = "research_report.md"
    with open(filename, "w") as f:
        f.write(f"# Research Report: {topic}\n\n{report}")

    print(f"\n✅ Saved to {filename}")