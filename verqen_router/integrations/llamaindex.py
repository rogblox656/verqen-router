"""LlamaIndex integration: VerQen as the query router for destination engines."""

from __future__ import annotations

from typing import Any

from verqen_router.core import RouteResult, Router

__all__ = ["LlamaIndexRouter", "as_index_tool"]


class LlamaIndexRouter:
    """Route queries to LlamaIndex query engines by VerQen verdict (pure logic).

    Args:
        router: A :class:`~verqen_router.core.Router` whose agent names match
            the keys of *destinations*.
        destinations: Mapping of agent name to an object with a ``query``
            method (usually a LlamaIndex ``QueryEngine``).
    """

    def __init__(self, router: Router, destinations: dict[str, Any] | None = None):
        self.router = router
        self.destinations = dict(destinations or {})

    def route(self, query: str, context: str = "") -> tuple[RouteResult, Any]:
        """Route *query* and return ``(result, destination_or_None)``."""
        result = self.router.route(task=query, context=context)
        return result, self.destinations.get(result.agent)

    def query(self, query: str, context: str = "") -> Any:
        """Route *query* and invoke the destination engine's ``query`` method."""
        result, engine = self.route(query, context)
        if engine is None:
            raise KeyError(f"no query engine mapped for agent {result.agent!r}")
        return engine.query(query)


def as_index_tool(router: Router, name: str = "verqen_router", description: str | None = None):
    """Expose the router as a LlamaIndex ``FunctionTool``.

    Raises ``ImportError`` with an install hint when LlamaIndex is missing.
    """
    try:
        from llama_index.core.tools import FunctionTool
    except ImportError as exc:
        raise ImportError(
            "LlamaIndex is required for as_index_tool(). "
            "Install it with: pip install verqen-router[llamaindex]"
        ) from exc

    def route_task(task: str) -> str:
        return router.route(task=task).agent

    return FunctionTool.from_defaults(
        fn=route_task,
        name=name,
        description=description
        or "Route a task to the best agent with the VerQen decision model.",
    )
