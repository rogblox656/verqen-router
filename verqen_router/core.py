"""Core of verqen-router: the Router class and its result types."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from verqen_router.conformal import confidence_of, is_ambiguous, prediction_set
from verqen_router.engine import load_engine

__all__ = ["AgentSpec", "RouteResult", "Router"]

ROUTING_QUESTION = "Which agent or tool should handle this task?"
MIN_AGENTS = 2
MAX_AGENTS = 64


@dataclass
class AgentSpec:
    """One routable agent or tool.

    Args:
        name: Unique identifier, also the value returned in ``RouteResult.agent``.
        description: What the agent does (shown to humans, not to the model).
        keywords: Words that strongly indicate this agent; used by
            :meth:`verqen_router.registry.AgentRegistry.quick_route`.
    """

    name: str
    description: str = ""
    keywords: list[str] = field(default_factory=list)


@dataclass
class RouteResult:
    """Verdict for a single routing decision."""

    agent: str
    confidence: float
    conformal_set: list[str]
    is_ambiguous: bool
    used_fallback: bool
    latency_ms: float


class Router:
    """Multi-agent task router backed by the VerQen decision model.

    One ``route`` call is exactly one ``engine.decide`` call over the agent
    name options, so routing stays in single-digit milliseconds no matter how
    many agents you register (2..64).

    Args:
        agents: 2..64 agent names (str) or :class:`AgentSpec` instances.
        checkpoint: Path to a VerQen checkpoint. Falls back to the
            ``VERQEN_CHECKPOINT`` environment variable, resolved lazily on
            first route.
        coverage: Target conformal coverage for the certified routing set.
        fallback_agent: Agent used when the engine is unavailable or raises.
            Without it, engine errors propagate to the caller.
        engine: Inject a pre-built engine or test double with ``decide``.
            Routing then needs no checkpoint at all.
    """

    def __init__(
        self,
        agents: list[AgentSpec | str],
        checkpoint: str | None = None,
        coverage: float = 0.95,
        fallback_agent: str | None = None,
        engine: Any | None = None,
    ):
        if isinstance(agents, str) or not isinstance(agents, (list, tuple)):
            raise TypeError("agents must be a list of names or AgentSpec instances")
        specs = [AgentSpec(a, a) if isinstance(a, str) else a for a in agents]
        names = [spec.name for spec in specs]
        if not (MIN_AGENTS <= len(names) <= MAX_AGENTS):
            raise ValueError(f"router needs {MIN_AGENTS} to {MAX_AGENTS} agents, got {len(names)}")
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("agent names must be non-empty strings")
        if len(set(names)) != len(names):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            raise ValueError(f"duplicate agent names: {duplicates}")
        if not 0.0 < coverage <= 1.0:
            raise ValueError(f"coverage must be in (0.0, 1.0], got {coverage!r}")
        if fallback_agent is not None and (
            not isinstance(fallback_agent, str) or not fallback_agent.strip()
        ):
            raise ValueError("fallback_agent must be a non-empty string")

        self.agents = specs
        self.agent_names = names
        self.coverage = float(coverage)
        self.fallback_agent = fallback_agent
        self._checkpoint = checkpoint
        self._engine = engine

    @property
    def engine(self) -> Any:
        """The engine, loaded lazily from ``checkpoint``/env on first use."""
        if self._engine is None:
            self._engine = load_engine(self._checkpoint)
        return self._engine

    def route(self, task: str, context: str = "") -> RouteResult:
        """Route *task* to the best agent with exactly one ``decide`` call.

        Raises ``ValueError`` for empty input. When the engine is missing or
        raises and ``fallback_agent`` is set, returns a fallback result
        instead of propagating the error.
        """
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        state = f"{context}\n\nTask: {task}" if context else task

        try:
            engine = self.engine
        except Exception:
            if self.fallback_agent is None:
                raise
            return self._fallback(0.0)

        start = time.perf_counter()
        try:
            decision = engine.decide(
                state,
                ROUTING_QUESTION,
                list(self.agent_names),
                question_type="choice",
            )
            selected = str(decision.get("selected", ""))
            if selected not in self.agent_names:
                raise ValueError(f"engine returned an unknown agent {selected!r}")
        except Exception:
            if self.fallback_agent is None:
                raise
            return self._fallback((time.perf_counter() - start) * 1000)

        conformal = prediction_set(decision)
        return RouteResult(
            agent=selected,
            confidence=confidence_of(decision),
            conformal_set=conformal,
            is_ambiguous=is_ambiguous(conformal),
            used_fallback=False,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    def route_multi(self, tasks: list[str], context: str = "") -> list[RouteResult]:
        """Route several tasks against shared *context* (one decide per task)."""
        if isinstance(tasks, str) or not isinstance(tasks, (list, tuple)):
            raise ValueError("tasks must be a list of task strings")
        return [self.route(task, context) for task in tasks]

    def _fallback(self, latency_ms: float) -> RouteResult:
        return RouteResult(
            agent=self.fallback_agent,  # type: ignore[arg-type]
            confidence=0.0,
            conformal_set=[self.fallback_agent],  # type: ignore[list-item]
            is_ambiguous=False,
            used_fallback=True,
            latency_ms=latency_ms,
        )
