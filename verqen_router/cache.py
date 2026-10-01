"""State cache: encode a shared context once, answer many routing questions.

Engines that expose ``encode_state`` and ``decide_on_state`` can encode a
state once and answer any number of questions against that one encoding, so
N questions on the same context cost 1 encode plus N cheap decisions.
Engines without those methods fall back to plain ``decide`` calls and the
results are identical either way.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

__all__ = ["StateCache"]


class StateCache:
    """LRU cache of encoded states around a VerQen engine.

    Args:
        engine: Object with ``decide`` and, optionally, ``encode_state`` and
            ``decide_on_state`` (the real VerQen engine exposes all three).
        max_entries: LRU capacity for encoded states (default 128).

    ``stats()`` reports hits, misses, encodes and the measured per-query
    costs: ``avg_encode_ms`` (cold path) and ``avg_reuse_ms`` (cost of one
    question answered against a cached encoding).
    """

    def __init__(self, engine: Any, max_entries: int = 128):
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.engine = engine
        self.max_entries = max_entries
        self._entries: OrderedDict[str, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.encodes = 0
        self._encode_ms = 0.0
        self._reuse_ms = 0.0
        self._reuse_queries = 0

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        """Drop every cached encoding and reset the counters."""
        self._entries.clear()
        self.hits = 0
        self.misses = 0
        self.encodes = 0
        self._encode_ms = 0.0
        self._reuse_ms = 0.0
        self._reuse_queries = 0

    @property
    def supports_state_encoding(self) -> bool:
        """True when the engine offers ``encode_state`` + ``decide_on_state``."""
        return callable(getattr(self.engine, "encode_state", None)) and callable(
            getattr(self.engine, "decide_on_state", None)
        )

    def encode(self, state: str) -> Any:
        """Encoded handle for *state*, computed at most once until evicted."""
        self._check_state(state)
        if not self.supports_state_encoding:
            raise RuntimeError("engine does not expose encode_state/decide_on_state")
        handle = self._entries.get(state)
        if handle is not None:
            self.hits += 1
            self._entries.move_to_end(state)
            return handle
        self.misses += 1
        start = time.perf_counter()
        handle = self.engine.encode_state(state)
        self._encode_ms += (time.perf_counter() - start) * 1000
        self.encodes += 1
        self._entries[state] = handle
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return handle

    def decide(
        self,
        state: str,
        question: str,
        options: list[str],
        question_type: str = "choice",
    ) -> dict:
        """One decision, reusing the cached encoding of *state* when possible."""
        self._check_state(state)
        if not self.supports_state_encoding:
            return self.engine.decide(state, question, options, question_type=question_type)
        handle = self.encode(state)
        start = time.perf_counter()
        answered = self.engine.decide_on_state(
            handle,
            [{"type": question_type, "text": question, "options": list(options)}],
        )
        self._record_reuse((time.perf_counter() - start) * 1000, 1)
        return answered[0]

    def decide_many(self, state: str, questions: list[dict]) -> list[dict]:
        """Answer several ``{"type", "text", "options"}`` questions against ONE encoding."""
        self._check_state(state)
        if not self.supports_state_encoding:
            return [
                self.engine.decide(
                    state,
                    question.get("text", ""),
                    question.get("options"),
                    question_type=question.get("type", "choice"),
                )
                for question in questions
            ]
        handle = self.encode(state)
        start = time.perf_counter()
        answered = self.engine.decide_on_state(handle, questions)
        self._record_reuse((time.perf_counter() - start) * 1000, len(questions))
        return list(answered)

    def stats(self) -> dict:
        """Cache counters and measured per-query costs in milliseconds."""
        return {
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "encodes": self.encodes,
            "state_caching": self.supports_state_encoding,
            "avg_encode_ms": round(self._encode_ms / self.encodes, 6) if self.encodes else 0.0,
            "avg_reuse_ms": (
                round(self._reuse_ms / self._reuse_queries, 6) if self._reuse_queries else 0.0
            ),
        }

    def _record_reuse(self, elapsed_ms: float, queries: int) -> None:
        self._reuse_ms += elapsed_ms
        self._reuse_queries += queries

    @staticmethod
    def _check_state(state: str) -> None:
        if not isinstance(state, str) or not state.strip():
            raise ValueError("state must be a non-empty string")
