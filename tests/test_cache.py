"""StateCache tests: LRU encoding reuse and plain-decide fallback (no GPU)."""

from __future__ import annotations

import pytest

from verqen_router.cache import StateCache


def _answer(question: dict) -> dict:
    options = list(question.get("options") or [])
    selected = options[0] if options else ""
    return {
        "type": question.get("type", "choice"),
        "selected": selected,
        "probabilities": {option: (1.0 if option == selected else 0.0) for option in options},
        "p_correct": 1.0,
        "confidence": 1.0,
        "margin": 1.0,
        "should_escalate": False,
        "prediction_set": [selected],
    }


class EncodingEngine:
    """Fake engine WITH encode_state/decide_on_state."""

    def __init__(self):
        self.encoded: list[str] = []
        self.state_decisions = 0
        self.plain_decisions = 0

    def decide(self, state, question, options=None, question_type="choice", cost=None):
        self.plain_decisions += 1
        return _answer({"type": question_type, "text": question, "options": options})

    def encode_state(self, state):
        self.encoded.append(state)
        return f"enc::{state}"

    def decide_on_state(self, handle, questions):
        self.state_decisions += 1
        assert str(handle).startswith("enc::")
        return [_answer(question) for question in questions]


class PlainEngine:
    """Fake engine WITHOUT the state-caching methods."""

    def __init__(self):
        self.decisions = 0

    def decide(self, state, question, options=None, question_type="choice", cost=None):
        self.decisions += 1
        return _answer({"type": question_type, "text": question, "options": options})


def test_encode_caches_per_state():
    engine = EncodingEngine()
    cache = StateCache(engine)
    first = cache.encode("ctx")
    second = cache.encode("ctx")
    assert first == second == "enc::ctx"
    assert engine.encoded == ["ctx"]
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["encodes"] == 1
    assert stats["state_caching"] is True
    assert stats["avg_encode_ms"] >= 0.0


def test_lru_eviction():
    engine = EncodingEngine()
    cache = StateCache(engine, max_entries=2)
    for state in ("a", "b", "c"):
        cache.encode(state)
    assert len(cache) == 2
    assert engine.encoded == ["a", "b", "c"]
    cache.encode("a")  # evicted, must re-encode
    assert len(engine.encoded) == 4
    assert cache.encodes == 4


def test_lru_hit_refreshes_entry():
    engine = EncodingEngine()
    cache = StateCache(engine, max_entries=2)
    cache.encode("a")
    cache.encode("b")
    cache.encode("a")  # hit: refreshes "a", so "b" is the eviction candidate
    cache.encode("c")
    cache.encode("b")  # evicted by "c", must re-encode (proves the hit refreshed "a")
    assert engine.encoded == ["a", "b", "c", "b"]


def test_decide_reuses_encoding():
    engine = EncodingEngine()
    cache = StateCache(engine)
    first = cache.decide("ctx", "Which agent?", ["x", "y"])
    second = cache.decide("ctx", "Which agent again?", ["x", "y"])
    assert first["selected"] == "x"
    assert second["selected"] == "x"
    assert engine.encoded == ["ctx"]  # one encode for two questions
    assert engine.state_decisions == 2
    assert engine.plain_decisions == 0
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["avg_reuse_ms"] >= 0.0


def test_decide_falls_back_without_state_support():
    engine = PlainEngine()
    cache = StateCache(engine)
    assert cache.supports_state_encoding is False
    result = cache.decide("ctx", "Which agent?", ["x", "y"])
    assert result["selected"] == "x"
    assert engine.decisions == 1
    stats = cache.stats()
    assert stats["state_caching"] is False
    assert stats["encodes"] == 0
    assert stats["hits"] == 0
    assert stats["misses"] == 0


def test_decide_many_one_encoding():
    engine = EncodingEngine()
    cache = StateCache(engine)
    questions = [
        {"type": "choice", "text": f"question {i}", "options": ["a", "b"]}
        for i in range(5)
    ]
    answers = cache.decide_many("ctx", questions)
    assert len(answers) == 5
    assert engine.encoded == ["ctx"]  # ONE encode for all five questions
    assert engine.state_decisions == 1
    assert cache.encodes == 1
    assert cache.stats()["avg_reuse_ms"] >= 0.0


def test_decide_many_falls_back_to_plain_decide():
    engine = PlainEngine()
    cache = StateCache(engine)
    questions = [{"type": "choice", "text": f"q{i}", "options": ["a"]} for i in range(3)]
    answers = cache.decide_many("ctx", questions)
    assert len(answers) == 3
    assert engine.decisions == 3
    assert cache.stats()["encodes"] == 0


def test_max_entries_validated():
    with pytest.raises(ValueError, match="max_entries"):
        StateCache(PlainEngine(), max_entries=0)


def test_empty_state_rejected():
    cache = StateCache(EncodingEngine())
    with pytest.raises(ValueError, match="state"):
        cache.encode("   ")
    with pytest.raises(ValueError, match="state"):
        cache.decide("", "q", ["a"])


def test_encode_without_support_raises():
    with pytest.raises(RuntimeError, match="encode_state"):
        StateCache(PlainEngine()).encode("ctx")


def test_clear_resets_counters():
    engine = EncodingEngine()
    cache = StateCache(engine)
    cache.encode("a")
    cache.decide("a", "q", ["x"])
    cache.clear()
    assert len(cache) == 0
    assert cache.stats() == {
        "entries": 0,
        "hits": 0,
        "misses": 0,
        "encodes": 0,
        "state_caching": True,
        "avg_encode_ms": 0.0,
        "avg_reuse_ms": 0.0,
    }
