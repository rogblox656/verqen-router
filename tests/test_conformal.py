"""Conformal set helper tests (pure functions, no model)."""

from __future__ import annotations

from verqen_router.conformal import confidence_of, is_ambiguous, prediction_set


def test_prediction_set_reads_key():
    decision = {"selected": "web_search", "prediction_set": ["web_search", "database_query"]}
    assert prediction_set(decision) == ["web_search", "database_query"]


def test_prediction_set_falls_back_to_selected():
    assert prediction_set({"selected": "web_search"}) == ["web_search"]


def test_prediction_set_empty_falls_back():
    assert prediction_set({"selected": "web_search", "prediction_set": []}) == ["web_search"]


def test_prediction_set_values_become_strings():
    decision = {"selected": "a", "prediction_set": ["a", 2]}
    assert prediction_set(decision) == ["a", "2"]


def test_is_ambiguous():
    assert is_ambiguous(["a", "b"]) is True
    assert is_ambiguous(["a"]) is False
    assert is_ambiguous([]) is False


def test_confidence_of_probabilities():
    decision = {"selected": "b", "probabilities": {"a": 0.6, "b": 0.4}}
    assert confidence_of(decision) == 0.4


def test_confidence_of_falls_back_to_flat_field():
    decision = {"selected": "a", "probabilities": {}, "confidence": 0.7}
    assert confidence_of(decision) == 0.7


def test_confidence_of_defaults_to_zero():
    assert confidence_of({}) == 0.0
    assert confidence_of({"selected": "a", "confidence": None}) == 0.0
