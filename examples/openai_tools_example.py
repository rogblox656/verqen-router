"""OpenAI tool routing with verqen-router: VerQen picks the tool, not the LLM.

Builds OpenAI function-tool schemas from the router's agents, routes each
task, and shows the chosen schema. Pure local inference: no OpenAI API key
and no network access needed to run it.

    python examples/openai_tools_example.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verqen_router import AgentSpec, Router  # noqa: E402
from verqen_router.integrations import agent_tool_schemas, select_tool  # noqa: E402

SPECS = [
    AgentSpec("web_search", "Search the web for current information"),
    AgentSpec("database_query", "Run read-only SQL against the warehouse"),
    AgentSpec("email_sender", "Send an email through the mail API"),
    AgentSpec("calculator", "Evaluate arithmetic expressions"),
]

TASKS = [
    "What was the revenue last quarter? Pull it from the warehouse.",
    "Send the invoice PDF to billing@acme.example",
    "What is 23 * 17 + 42?",
]


def main() -> int:
    if not os.environ.get("VERQEN_CHECKPOINT"):
        print(
            "Set VERQEN_CHECKPOINT=/path/to/verqen-checkpoint first.",
            file=sys.stderr,
        )
        return 1

    router = Router(agents=SPECS, fallback_agent="web_search")
    tools = agent_tool_schemas(router.agents)

    print("OpenAI tool schemas built from the router's agents:")
    for tool in tools:
        print(f"  - {tool['function']['name']}: {tool['function']['description']}")

    for task in TASKS:
        result, tool = select_tool(router, tools, task)
        chosen = tool["function"]["name"] if tool else "(no matching tool)"
        print(f"\nTask:     {task}")
        print(
            f"Tool:     {chosen}  confidence={result.confidence:.2f}  "
            f"set={result.conformal_set}  {result.latency_ms:.2f} ms"
        )
        print("Tip: pass tools= and a chat client to route_completion() to force "
              "the chosen tool on an OpenAI call.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
