"""AgentRegistry tests: lifecycle plus keyword quick routing (no model)."""

from __future__ import annotations

import pytest

from verqen_router import AgentSpec
from verqen_router.registry import AgentRegistry


@pytest.fixture
def registry():
    return AgentRegistry(
        [
            AgentSpec("web_search", "Searches the web", ["search", "lookup"]),
            AgentSpec("database_query", "Runs SQL queries", ["sql", "database"]),
            AgentSpec("email_sender", "Sends email", ["email"]),
        ]
    )


def test_register_and_lookup():
    registry = AgentRegistry()
    spec = registry.register("web_search", description="Searches the web", keywords=["search"])
    assert isinstance(spec, AgentSpec)
    assert registry.lookup("web_search") is spec
    assert registry.lookup("web_search").keywords == ["search"]


def test_init_accepts_mixed_agents():
    registry = AgentRegistry(["web_search", AgentSpec("email_sender", "Sends email")])
    assert registry.list_names() == ["web_search", "email_sender"]


def test_duplicate_register_raises(registry):
    with pytest.raises(ValueError, match="already registered"):
        registry.register("web_search")


def test_spec_with_extras_raises(registry):
    with pytest.raises(ValueError, match="not both"):
        registry.register(AgentSpec("calendar_api", "Books meetings"), keywords=["meeting"])


def test_blank_name_raises(registry):
    with pytest.raises(ValueError, match="non-empty"):
        registry.register("   ")


def test_lookup_unknown_raises(registry):
    with pytest.raises(KeyError, match="unknown agent"):
        registry.lookup("ghost_agent")


def test_unregister(registry):
    removed = registry.unregister("email_sender")
    assert removed.name == "email_sender"
    assert "email_sender" not in registry
    assert registry.list_names() == ["web_search", "database_query"]
    with pytest.raises(KeyError):
        registry.unregister("email_sender")


def test_list_names_preserves_order(registry):
    assert registry.list_names() == ["web_search", "database_query", "email_sender"]
    assert len(registry) == 3


def test_quick_route_keyword_match(registry):
    result = registry.quick_route("search the web for python news")
    assert result.agent == "web_search"
    assert result.used_fallback is True
    assert result.is_ambiguous is False
    assert result.conformal_set == ["web_search"]
    # two hits ("search" keyword + "search"/"web" name tokens): 0.55 + 0.15.
    assert result.confidence == pytest.approx(0.70)
    assert result.latency_ms >= 0


def test_quick_route_matches_name_tokens():
    registry = AgentRegistry(["database_query", "email_sender"])
    result = registry.quick_route("look up the customer database")
    assert result.agent == "database_query"
    assert result.confidence == pytest.approx(0.55)


def test_quick_route_tie_is_ambiguous(registry):
    result = registry.quick_route("run a search against the database")
    assert result.is_ambiguous is True
    assert set(result.conformal_set) == {"web_search", "database_query"}
    assert result.agent in result.conformal_set


def test_quick_route_no_signal(registry):
    result = registry.quick_route("hello world")
    assert result.confidence == 0.0
    assert result.is_ambiguous is True
    assert set(result.conformal_set) == {"web_search", "database_query", "email_sender"}
    assert result.agent == "web_search"  # first registered wins
    assert result.used_fallback is True


def test_quick_route_uses_context(registry):
    result = registry.quick_route("do it", context="the user asked to send an email")
    assert result.agent == "email_sender"


def test_quick_route_empty_registry_raises():
    with pytest.raises(ValueError, match="no agents"):
        AgentRegistry().quick_route("anything")


def test_quick_route_empty_task_raises(registry):
    with pytest.raises(ValueError, match="non-empty"):
        registry.quick_route("")
