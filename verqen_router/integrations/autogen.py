"""AutoGen integration: let VerQen pick which AutoGen agent handles a task."""

from __future__ import annotations

from typing import Any, Iterable

from verqen_router.core import AgentSpec, RouteResult, Router

__all__ = ["AutoGenRouter", "build_autogen_agents"]


class AutoGenRouter:
    """Hand tasks to AutoGen agents by VerQen verdict (pure logic, no autogen import).

    Args:
        router: A :class:`~verqen_router.core.Router` whose agent names match
            the keys of *agents*.
        agents: Mapping of agent name to an already-built AutoGen agent.
    """

    def __init__(self, router: Router, agents: dict[str, Any] | None = None):
        self.router = router
        self.agents = dict(agents or {})

    def handoff(self, task: str, context: str = "") -> tuple[RouteResult, Any]:
        """Route *task* and return ``(result, autogen_agent_or_None)``."""
        result = self.router.route(task=task, context=context)
        return result, self.agents.get(result.agent)


def build_autogen_agents(
    specs: Iterable[AgentSpec | str],
    llm_config: Any | None = None,
) -> dict[str, Any]:
    """Build one AutoGen ``ConversableAgent`` per spec, keyed by agent name.

    ``llm_config`` is passed through untouched (``None`` becomes ``False``,
    meaning a hand-written responder with no LLM of its own).

    Raises ``ImportError`` with an install hint when AutoGen is missing.
    """
    try:
        from autogen import ConversableAgent
    except ImportError as exc:
        raise ImportError(
            "AutoGen is required for build_autogen_agents(). "
            "Install it with: pip install verqen-router[autogen] (installs pyautogen)"
        ) from exc
    built: dict[str, Any] = {}
    for spec in specs:
        normalized = AgentSpec(spec, spec) if isinstance(spec, str) else spec
        built[normalized.name] = ConversableAgent(
            name=normalized.name,
            llm_config=llm_config if llm_config is not None else False,
            human_input_mode="NEVER",
        )
    return built
