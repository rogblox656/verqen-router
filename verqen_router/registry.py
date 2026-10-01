"""AgentRegistry: register, look up and keyword-route agents without a model."""

from __future__ import annotations

import re
import time
from typing import Iterable, Sequence

from verqen_router.core import AgentSpec, RouteResult

__all__ = ["AgentRegistry"]


def _name_tokens(name: str) -> list[str]:
    """Significant tokens of an agent name (``database_query`` -> database, query)."""
    return [token for token in re.split(r"[^a-z0-9]+", name.lower()) if len(token) >= 3]


class AgentRegistry:
    """Registry of routable agents with a keyword-based fallback router.

    ``quick_route`` scores each agent by keyword hits in the task text and
    needs no engine, no checkpoint and no GPU: useful for cold starts, tests,
    and as a documented last-resort path when the model is unavailable.
    """

    def __init__(self, agents: Iterable[AgentSpec | str] = ()):
        self._agents: dict[str, AgentSpec] = {}
        for agent in agents:
            self.register(agent)

    def register(
        self,
        agent: AgentSpec | str,
        description: str = "",
        keywords: Sequence[str] = (),
    ) -> AgentSpec:
        """Add *agent* to the registry. Raises ``ValueError`` on duplicates."""
        if isinstance(agent, AgentSpec):
            if description or keywords:
                raise ValueError(
                    "pass either an AgentSpec or a name with description/keywords, not both"
                )
            spec = agent
        else:
            if not isinstance(agent, str) or not agent.strip():
                raise ValueError("agent name must be a non-empty string")
            spec = AgentSpec(agent, description or agent, list(keywords))
        if spec.name in self._agents:
            raise ValueError(f"agent already registered: {spec.name!r}")
        self._agents[spec.name] = spec
        return spec

    def lookup(self, name: str) -> AgentSpec:
        """Return the :class:`AgentSpec` for *name* (raises ``KeyError``)."""
        if name not in self._agents:
            raise KeyError(f"unknown agent {name!r}; registered: {self.list_names()}")
        return self._agents[name]

    def unregister(self, name: str) -> AgentSpec:
        """Remove *name* and return its spec (raises ``KeyError`` if absent)."""
        spec = self.lookup(name)
        del self._agents[name]
        return spec

    def list_names(self) -> list[str]:
        """Registered agent names in registration order."""
        return list(self._agents)

    def __contains__(self, name: object) -> bool:
        return name in self._agents

    def __len__(self) -> int:
        return len(self._agents)

    def quick_route(self, task: str, context: str = "") -> RouteResult:
        """Keyword-based routing: score agents by keyword hits, no model call.

        Every registered keyword and every significant name token (3+ chars)
        counts once as a hit. Tied top scorers all appear in the conformal
        set; the first registered wins the ``agent`` slot. No hits at all
        means genuine uncertainty: confidence 0.0 and the whole registry in
        the set. The result always has ``used_fallback=True``.
        """
        if not self._agents:
            raise ValueError("registry has no agents to route to")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        text = f"{context}\n{task}".lower() if context else task.lower()

        start = time.perf_counter()
        scores: dict[str, int] = {}
        for name, spec in self._agents.items():
            needles = {kw.lower() for kw in spec.keywords if kw}
            needles.update(_name_tokens(name))
            scores[name] = sum(1 for needle in needles if needle in text)

        top = max(scores.values())
        winners = [name for name, score in scores.items() if score == top]
        if top == 0:
            confidence = 0.0
        else:
            confidence = min(0.99, 0.55 + 0.15 * (top - 1))
        return RouteResult(
            agent=winners[0],
            confidence=confidence,
            conformal_set=winners,
            is_ambiguous=len(winners) > 1,
            used_fallback=True,
            latency_ms=(time.perf_counter() - start) * 1000,
        )
