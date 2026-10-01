"""CrewAI + verqen-router: let VerQen assign each task to a CrewAI agent.

Runnable with a local checkpoint (``VERQEN_CHECKPOINT``). CrewAI itself is
optional: without it, the example still prints the routing decisions.

    python examples/crewai_example.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verqen_router import Router  # noqa: E402
from verqen_router.integrations import CrewAIRouter  # noqa: E402

AGENTS = ["web_search", "database_query", "email_sender", "calendar_api", "document_reader"]

TASKS = [
    "Find the latest quarterly revenue figures for Acme Corp",
    "Email the meeting summary to the design team",
    "Book a room for the architecture review on Wednesday",
]


def main() -> int:
    if not os.environ.get("VERQEN_CHECKPOINT"):
        print(
            "Set VERQEN_CHECKPOINT=/path/to/verqen-checkpoint first.",
            file=sys.stderr,
        )
        return 1

    router = Router(agents=AGENTS, fallback_agent="web_search")

    crew_agents = None
    try:
        from verqen_router.integrations import build_crewai_agents

        crew_agents = build_crewai_agents(router.agents)
    except ImportError as exc:
        print(f"note: {exc}")
        print("Continuing with routing decisions only.")

    crew_router = CrewAIRouter(router, agents=dict(zip(AGENTS, crew_agents or [])))

    for task in TASKS:
        result, agent = crew_router.assign(task)
        print(f"\nTask:     {task}")
        print(
            f"Routed:   {result.agent}  confidence={result.confidence:.2f}  "
            f"set={result.conformal_set}  {result.latency_ms:.2f} ms"
        )
        if agent is not None:
            print(f"CrewAI:   role={agent.role!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
