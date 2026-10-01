"""Conformal routing sets: certified coverage helpers for route decisions.

The VerQen engine returns a ``prediction_set`` alongside the selected agent.
That set is the conformal prediction set: at the model's certified coverage
level (95%), the true agent is inside the set that often. A set with more
than one entry means the router is honestly uncertain between those agents.
"""

from __future__ import annotations

from typing import Any, Sequence

__all__ = ["prediction_set", "is_ambiguous", "confidence_of"]


def prediction_set(decision: dict[str, Any]) -> list[str]:
    """Certified conformal set from a ``decide()`` result.

    Reads the engine's ``prediction_set`` key (older builds may omit it, and
    some engines only report ``selected``), so the result is never empty.
    """
    selected = str(decision.get("selected", ""))
    values = decision.get("prediction_set")
    if not values:
        return [selected]
    return [str(value) for value in values]


def is_ambiguous(conformal_set: Sequence[str]) -> bool:
    """True when the certified set names more than one agent."""
    return len(conformal_set) > 1


def confidence_of(decision: dict[str, Any]) -> float:
    """Calibrated confidence for the selected agent.

    Uses the selected entry of ``probabilities`` when present, otherwise the
    flat ``confidence`` field the engine reports.
    """
    selected = str(decision.get("selected", ""))
    probabilities = decision.get("probabilities") or {}
    if selected in probabilities:
        return float(probabilities[selected])
    return float(decision.get("confidence", 0.0) or 0.0)
