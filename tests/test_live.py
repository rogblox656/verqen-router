"""Live tests against a real VerQen checkpoint.

Skipped unless ``VERQEN_CHECKPOINT`` is set. Run with::

    pytest -m live
"""

from __future__ import annotations

import os
import time

import pytest

from verqen_router import Router

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("VERQEN_CHECKPOINT"),
        reason="VERQEN_CHECKPOINT not set",
    ),
]

AGENTS = [
    "code_executor",
    "web_search",
    "database_query",
    "email_sender",
    "calendar_api",
    "document_reader",
    "translator",
    "calculator",
]


@pytest.fixture(scope="module")
def live_router():
    return Router(agents=AGENTS)


def test_route_returns_known_agent(live_router):
    result = live_router.route("Run this Python script and print its output.")
    assert result.agent in AGENTS
    assert result.used_fallback is False
    assert 0.0 <= result.confidence <= 1.0
    assert result.latency_ms > 0


def test_conformal_set_within_options(live_router):
    result = live_router.route("Find the latest news about electric vehicles.")
    assert result.conformal_set
    assert set(result.conformal_set) <= set(AGENTS)
    assert result.agent in result.conformal_set
    assert result.is_ambiguous == (len(result.conformal_set) > 1)


def test_route_multi_live(live_router):
    results = live_router.route_multi(
        ["Translate this sentence to French.", "Add a meeting for Friday at 3pm."],
        context="User is on a mobile device.",
    )
    assert len(results) == 2
    assert all(result.agent in AGENTS for result in results)
    assert all(result.used_fallback is False for result in results)


def test_latency_budget_eager(live_router):
    """Warm call budget: eager mode is ~68ms on an RTX 5060 Ti; allow 2s CI slack."""
    live_router.route("warmup routing question for timing")
    start = time.perf_counter()
    live_router.route("What is 15% of 240?")
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 2000
