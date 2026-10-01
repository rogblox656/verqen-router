"""CrewAI integration: let VerQen pick which CrewAI agent gets each task."""

from __future__ import annotations

from typing import Any, Iterable

from verqen_router.core import AgentSpec, RouteResult, Router

__all__ = ["CrewAIRouter", "build_crewai_agents"]


class CrewAIRouter:
    """Delegate tasks to CrewAI agents by VerQen verdict (pure logic, no crewai import).

    Args:
        router: A :class:`~verqen_router.core.Router` whose agent names match
            the keys of *agents*.
        agents: Mapping of agent name to an already-built CrewAI ``Agent``.
    """

    def __init__(self, router: Router, agents: dict[str, Any] | None = None):
        self.router = router
        self.agents = dict(agents or {})

    def assign(self, task: str, context: str = "") -> tuple[RouteResult, Any]:
        """Route *task* and return ``(result, crewai_agent_or_None)``."""
        result = self.router.route(task=task, context=context)
        return result, self.agents.get(result.agent)

    def assign_all(self, tasks: list[str], context: str = "") -> list[tuple[RouteResult, Any]]:
        """Route several tasks against shared *context*."""
        return [self.assign(task, context) for task in tasks]


def build_crewai_agents(
    specs: Iterable[AgentSpec | str],
    backstory: str | None = None,
) -> list[Any]:
    """Build one CrewAI ``Agent`` per spec (role = agent name, goal = description).

    Raises ``ImportError`` with an install hint when CrewAI is missing.
    """
    try:
        from crewai import Agent
    except ImportError as exc:
        raise ImportError(
            "CrewAI is required for build_crewai_agents(). "
            "Install it with: pip install verqen-router[crewai]"
        ) from exc
    built = []
    for spec in specs:
        normalized = AgentSpec(spec, spec) if isinstance(spec, str) else spec
        kwargs: dict[str, Any] = {
            "role": normalized.name,
            "goal": normalized.description or f"Handle tasks routed to {normalized.name}",
        }
        if backstory:
            kwargs["backstory"] = backstory
        built.append(Agent(**kwargs))
    return built
