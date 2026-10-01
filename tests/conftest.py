"""Shared test fixtures for verqen-router.

``FakeEngine`` mimics the real ``VerqenInferenceEngine.decide`` output
shape so the whole suite runs without a GPU, torch or a checkpoint.
"""

from __future__ import annotations

import pytest

from verqen_router import Router

DEFAULT_AGENTS = ["web_search", "database_query", "email_sender"]


class FakeEngine:
    """Deterministic stand-in for the VerQen engine.

    Args:
        selected: Answer for the routing question. When it is not one of the
            router's agent options, the first option is used.
        prediction_set: Override for ``prediction_set`` (default: ``[selected]``).
        confidence: Probability mass placed on the selected agent.
        error: When set, ``decide`` raises ``RuntimeError(error)``.
    """

    def __init__(
        self,
        selected: str = "web_search",
        prediction_set: list[str] | None = None,
        confidence: float = 0.9,
        error: str | None = None,
    ):
        self.selected = selected
        self.prediction_set = prediction_set
        self.confidence = confidence
        self.error = error
        self.calls: list[dict] = []

    def _answer(self, options: list[str], question_type: str) -> dict:
        selected = self.selected if self.selected in options else (options[0] if options else "")
        rest = (1.0 - self.confidence) / max(1, len(options) - 1) if len(options) > 1 else 0.0
        probabilities = {
            option: (self.confidence if option == selected else rest) for option in options
        }
        return {
            "type": question_type,
            "selected": selected,
            "probabilities": probabilities,
            "p_correct": self.confidence,
            "margin": round(self.confidence - rest, 6),
            "confidence": self.confidence,
            "should_escalate": False,
            "prediction_set": (
                list(self.prediction_set) if self.prediction_set is not None else [selected]
            ),
        }

    def decide(self, state, question, options=None, question_type="choice", cost=None):
        options = list(options or [])
        self.calls.append(
            {
                "state": state,
                "question": question,
                "options": options,
                "question_type": question_type,
            }
        )
        if self.error is not None:
            raise RuntimeError(self.error)
        return self._answer(options, question_type)

    def decide_multi(self, state: str, questions: list[dict]) -> list[dict]:
        answers = []
        for question in questions:
            self.calls.append(
                {
                    "state": state,
                    "question": question.get("text", ""),
                    "options": list(question.get("options") or []),
                    "question_type": question.get("type", "choice"),
                }
            )
            if self.error is not None:
                raise RuntimeError(self.error)
            answers.append(
                self._answer(
                    list(question.get("options") or []),
                    question.get("type", "choice"),
                )
            )
        return answers


@pytest.fixture
def fake_engine_cls():
    return FakeEngine


@pytest.fixture
def make_router():
    """Factory: ``make_router({"selected": "email_sender"}, agents=[...])``."""

    def _make(engine_kwargs: dict | None = None, **router_kwargs) -> Router:
        router_kwargs.setdefault("agents", list(DEFAULT_AGENTS))
        return Router(engine=FakeEngine(**(engine_kwargs or {})), **router_kwargs)

    return _make
