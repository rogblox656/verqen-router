"""LangChain integration: VerQen as the router for a multi-route chain."""

from __future__ import annotations

from typing import Any

from verqen_router.core import RouteResult, Router

__all__ = ["VerQenRouterChain", "as_runnable"]


class VerQenRouterChain:
    """Drop-in stand-in for LangChain's RouterChain: pure logic, no LangChain import.

    ``route`` returns LangChain's router shape, so a MultiRouteChain-style
    dispatcher can consume it directly::

        {"destination": "web_search", "next_inputs": {...}}

    Args:
        router: A :class:`~verqen_router.core.Router`.
        destinations: Optional mapping of agent name to the downstream chain
            that agent dispatches to.
        input_key: Which input field carries the task text.
    """

    def __init__(
        self,
        router: Router,
        destinations: dict[str, Any] | None = None,
        input_key: str = "input",
    ):
        self.router = router
        self.destinations = dict(destinations or {})
        self.input_key = input_key

    def select(self, inputs: dict[str, Any]) -> RouteResult:
        """Route the task in ``inputs[input_key]`` (context from ``inputs["context"]``)."""
        if not isinstance(inputs, dict):
            raise ValueError("inputs must be a dict")
        task = inputs.get(self.input_key)
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"inputs must contain a non-empty {self.input_key!r} field")
        context = inputs.get("context") or ""
        if not isinstance(context, str):
            raise ValueError("inputs['context'] must be a string")
        return self.router.route(task=task, context=context)

    def route(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """LangChain RouterChain output: ``destination`` plus ``next_inputs``."""
        result = self.select(inputs)
        return {"destination": result.agent, "next_inputs": inputs}

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """LangChain-style ``invoke`` alias for :meth:`route`."""
        return self.route(inputs)


def as_runnable(chain: VerQenRouterChain):
    """Wrap *chain* in a LangChain ``RunnableLambda``.

    Raises ``ImportError`` with an install hint when LangChain is missing.
    """
    try:
        from langchain_core.runnables import RunnableLambda
    except ImportError:
        try:  # LangChain < 0.2
            from langchain.schema.runnable import RunnableLambda
        except ImportError as exc:
            raise ImportError(
                "LangChain is required for as_runnable(). "
                "Install it with: pip install verqen-router[langchain]"
            ) from exc
    return RunnableLambda(chain.route)
