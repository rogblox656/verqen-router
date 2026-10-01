"""Core routing behaviour tests (no GPU, no checkpoint)."""

from __future__ import annotations

import pytest

from verqen_router import AgentSpec, Router
from verqen_router.core import MAX_AGENTS, ROUTING_QUESTION

AGENTS = ["web_search", "database_query", "email_sender"]


class OddEngine:
    """Engine double that answers with a name outside the option list."""

    def decide(self, state, question, options=None, question_type="choice", cost=None):
        return {
            "type": "choice",
            "selected": "ghost_agent",
            "probabilities": {"ghost_agent": 1.0},
            "confidence": 1.0,
            "prediction_set": ["ghost_agent"],
        }


def test_empty_task_raises(make_router):
    router = make_router()
    with pytest.raises(ValueError, match="non-empty"):
        router.route("")
    with pytest.raises(ValueError, match="non-empty"):
        router.route("   \n  ")


def test_route_returns_selected_agent(make_router):
    result = make_router().route("look up the weather in Tokyo")
    assert result.agent == "web_search"
    assert result.used_fallback is False
    assert result.confidence == pytest.approx(0.9)
    assert result.conformal_set == ["web_search"]
    assert result.is_ambiguous is False
    assert result.latency_ms >= 0


def test_route_is_one_decide_call(fake_engine_cls):
    engine = fake_engine_cls()
    router = Router(engine=engine, agents=AGENTS)
    router.route("send the quarterly report", context="user is a manager")
    assert len(engine.calls) == 1
    call = engine.calls[0]
    assert call["state"] == "user is a manager\n\nTask: send the quarterly report"
    assert call["question"] == ROUTING_QUESTION
    assert call["options"] == AGENTS
    assert call["question_type"] == "choice"


def test_route_state_without_context_is_task(fake_engine_cls):
    engine = fake_engine_cls()
    router = Router(engine=engine, agents=AGENTS)
    router.route("just the task")
    assert engine.calls[0]["state"] == "just the task"


def test_conformal_set_from_prediction_set(make_router):
    result = make_router(
        {"prediction_set": ["web_search", "database_query"]}
    ).route("search for the revenue table")
    assert result.conformal_set == ["web_search", "database_query"]
    assert result.is_ambiguous is True
    assert result.agent == "web_search"


def test_confidence_comes_from_probabilities(make_router):
    result = make_router({"confidence": 0.42}).route("anything at all")
    assert result.confidence == pytest.approx(0.42)


def test_three_agent_option_list(fake_engine_cls):
    engine = fake_engine_cls(selected="email_sender")
    router = Router(engine=engine, agents=AGENTS)
    assert router.agent_names == AGENTS
    result = router.route("notify the team by email")
    assert result.agent == "email_sender"
    assert engine.calls[0]["options"] == AGENTS


def test_sixtyfour_agent_option_list(fake_engine_cls):
    names = [f"agent_{i:02d}" for i in range(MAX_AGENTS)]
    engine = fake_engine_cls(selected="agent_63")
    router = Router(engine=engine, agents=names)
    assert len(router.agent_names) == MAX_AGENTS
    result = router.route("handle the task")
    assert result.agent == "agent_63"
    assert engine.calls[0]["options"] == names


def test_below_minimum_agents_raise(fake_engine_cls):
    with pytest.raises(ValueError, match="2 to 64 agents"):
        Router(engine=fake_engine_cls(), agents=["only_one"])


def test_above_maximum_agents_raise(fake_engine_cls):
    names = [f"agent_{i}" for i in range(MAX_AGENTS + 1)]
    with pytest.raises(ValueError, match="2 to 64 agents"):
        Router(engine=fake_engine_cls(), agents=names)


def test_duplicate_agent_names_raise(fake_engine_cls):
    with pytest.raises(ValueError, match="duplicate"):
        Router(engine=fake_engine_cls(), agents=["web_search", "web_search"])


def test_blank_agent_name_raises(fake_engine_cls):
    with pytest.raises(ValueError, match="non-empty"):
        Router(engine=fake_engine_cls(), agents=["web_search", "  "])


def test_agents_must_be_a_list(fake_engine_cls):
    with pytest.raises(TypeError, match="list"):
        Router(engine=fake_engine_cls(), agents="web_search")


def test_coverage_validated(fake_engine_cls):
    with pytest.raises(ValueError, match="coverage"):
        Router(engine=fake_engine_cls(), agents=AGENTS, coverage=0.0)
    with pytest.raises(ValueError, match="coverage"):
        Router(engine=fake_engine_cls(), agents=AGENTS, coverage=1.5)
    router = Router(engine=fake_engine_cls(), agents=AGENTS, coverage=1.0)
    assert router.coverage == 1.0


def test_fallback_agent_validated(fake_engine_cls):
    with pytest.raises(ValueError, match="fallback_agent"):
        Router(engine=fake_engine_cls(), agents=AGENTS, fallback_agent="")


def test_string_agents_become_specs(fake_engine_cls):
    router = Router(engine=fake_engine_cls(), agents=AGENTS)
    assert all(isinstance(spec, AgentSpec) for spec in router.agents)
    assert [spec.description for spec in router.agents] == AGENTS
    assert all(spec.keywords == [] for spec in router.agents)


def test_agentspec_passthrough(fake_engine_cls):
    spec = AgentSpec("web_search", "Searches the web", ["search"])
    router = Router(engine=fake_engine_cls(), agents=[spec, "email_sender"])
    assert router.agents[0] is spec
    assert router.agents[0].keywords == ["search"]


def test_engine_injected_without_checkpoint(make_router, monkeypatch):
    monkeypatch.delenv("VERQEN_CHECKPOINT", raising=False)
    result = make_router().route("still routes")
    assert result.used_fallback is False
    assert result.agent == "web_search"


def test_fallback_when_engine_raises(fake_engine_cls):
    engine = fake_engine_cls(error="engine exploded")
    router = Router(engine=engine, agents=AGENTS, fallback_agent="email_sender")
    result = router.route("anything")
    assert result.used_fallback is True
    assert result.agent == "email_sender"
    assert result.confidence == 0.0
    assert result.conformal_set == ["email_sender"]
    assert result.is_ambiguous is False
    assert result.latency_ms >= 0


def test_engine_error_without_fallback_propagates(fake_engine_cls):
    engine = fake_engine_cls(error="engine exploded")
    router = Router(engine=engine, agents=AGENTS)
    with pytest.raises(RuntimeError, match="engine exploded"):
        router.route("anything")


def test_fallback_when_no_engine(monkeypatch):
    monkeypatch.delenv("VERQEN_CHECKPOINT", raising=False)
    router = Router(agents=AGENTS, fallback_agent="web_search")
    result = router.route("no engine configured")
    assert result.used_fallback is True
    assert result.agent == "web_search"


def test_no_engine_no_fallback_raises(monkeypatch):
    monkeypatch.delenv("VERQEN_CHECKPOINT", raising=False)
    router = Router(agents=AGENTS)
    with pytest.raises(ValueError, match="VERQEN_CHECKPOINT"):
        router.route("no engine configured")


def test_unknown_selected_falls_back():
    router = Router(engine=OddEngine(), agents=AGENTS, fallback_agent="web_search")
    result = router.route("anything")
    assert result.used_fallback is True
    assert result.agent == "web_search"


def test_unknown_selected_without_fallback_raises():
    router = Router(engine=OddEngine(), agents=AGENTS)
    with pytest.raises(ValueError, match="unknown agent"):
        router.route("anything")


def test_route_multi(fake_engine_cls):
    engine = fake_engine_cls()
    router = Router(engine=engine, agents=AGENTS)
    results = router.route_multi(
        ["first task", "second task"], context="shared context"
    )
    assert len(results) == 2
    assert all(result.agent == "web_search" for result in results)
    assert all(result.used_fallback is False for result in results)
    states = [call["state"] for call in engine.calls]
    assert states == [
        "shared context\n\nTask: first task",
        "shared context\n\nTask: second task",
    ]


def test_route_multi_empty_list(make_router):
    assert make_router().route_multi([]) == []


def test_route_multi_rejects_bare_string(make_router):
    with pytest.raises(ValueError, match="list of task strings"):
        make_router().route_multi("not a list")
