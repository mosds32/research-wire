import os
import sys
import json
from mistralai.client import Mistral
from ddgs import DDGS
from dotenv import load_dotenv

load_dotenv()  # reads the .env file and loads variables into the environment

API_KEY = os.environ.get("MISTRAL_API_KEY")
MODEL = "mistral-large-latest"  # or "mistral-small-latest" for a cheaper/faster option

client = Mistral(api_key=API_KEY)


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


# Mistral uses the same OpenAI-style tool schema
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
    """Send a chat completion request to the Mistral AI API."""
    response = client.chat.complete(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        max_tokens=2000,
    )
    return response


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
        response = call_model(messages)
        choice = response.choices[0].message

        messages.append(
            {
                "role": "assistant",
                "content": choice.content,
                "tool_calls": choice.tool_calls,
            }
        )

        tool_calls = choice.tool_calls

        # No tool calls -> model is giving its final answer
        if not tool_calls:
            content = choice.content
            if content and content.strip():
                return content.strip()

            # The model sometimes stops without writing the report.
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
            retry_response = call_model(messages)
            retry_content = (retry_response.choices[0].message.content or "").strip()
            if retry_content:
                return retry_content

            return "(model returned empty content twice — try increasing max_tokens or rerunning)"

        # Otherwise, execute each requested tool call
        for tc in tool_calls:
            fn_name = tc.function.name

            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": fn_name,
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
                    "tool_call_id": tc.id,
                    "name": fn_name,
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
            "Missing MISTRAL_API_KEY. Set it as an environment variable "
            "before running this script:\n"
            "  export MISTRAL_API_KEY='your-key-here'\n"
            "Get a key from console.mistral.ai → API Keys."
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