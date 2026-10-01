"""AutoGen + verqen-router: hand each task to the AutoGen agent VerQen picks.

Runnable with a local checkpoint (``VERQEN_CHECKPOINT``). AutoGen itself is
optional: without it, the example still prints the routing decisions.

    python examples/autogen_example.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verqen_router import Router  # noqa: E402
from verqen_router.integrations import AutoGenRouter  # noqa: E402

AGENTS = ["code_executor", "web_search", "document_reader", "calculator"]

TASKS = [
    "Run this Python script and capture the output",
    "Look up the syntax for a Python list comprehension",
    "Calculate the compound interest on 10000 over 3 years",
]


def main() -> int:
    if not os.environ.get("VERQEN_CHECKPOINT"):
        print(
            "Set VERQEN_CHECKPOINT=/path/to/verqen-checkpoint first.",
            file=sys.stderr,
        )
        return 1

    router = Router(agents=AGENTS, fallback_agent="web_search")

    gen_agents = None
    try:
        from verqen_router.integrations import build_autogen_agents

        gen_agents = build_autogen_agents(router.agents)
    except ImportError as exc:
        print(f"note: {exc}")
        print("Continuing with routing decisions only.")

    gen_router = AutoGenRouter(router, agents=gen_agents or {})

    for task in TASKS:
        result, agent = gen_router.handoff(task)
        print(f"\nTask:     {task}")
        print(
            f"Routed:   {result.agent}  confidence={result.confidence:.2f}  "
            f"set={result.conformal_set}  {result.latency_ms:.2f} ms"
        )
        if agent is not None:
            print(f"AutoGen:  {agent.name!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
