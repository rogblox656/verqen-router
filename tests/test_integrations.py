"""Integration wrappers: import safety plus pure routing logic.

No agent framework is installed in the test environment, so every test that
needs one is guarded with ``importorskip`` (skipped here) and the missing
frameworks' ImportError hints are asserted instead.
"""

from __future__ import annotations

import importlib.util

import pytest

from verqen_router import AgentSpec, integrations
from verqen_router.integrations import (
    AutoGenRouter,
    CrewAIRouter,
    LlamaIndexRouter,
    VerQenRouterChain,
    agent_tool_schemas,
    route_completion,
    select_tool,
    tool_call_task,
)

AGENTS = ["web_search", "database_query", "email_sender"]

_HAS_LANGCHAIN = importlib.util.find_spec("langchain_core") is not None
_HAS_CREWAI = importlib.util.find_spec("crewai") is not None
_HAS_AUTOGEN = importlib.util.find_spec("autogen") is not None
_HAS_LLAMAINDEX = importlib.util.find_spec("llama_index") is not None
_HAS_OPENAI = importlib.util.find_spec("openai") is not None


# ---------------------------------------------------------------- import safety


def test_every_export_is_importable_without_frameworks():
    for name in integrations.__all__:
        assert callable(getattr(integrations, name)), name


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError, match="does_not_exist"):
        getattr(integrations, "does_not_exist")


# ---------------------------------------------------------------- langchain


def test_langchain_chain_route(make_router):
    chain = VerQenRouterChain(make_router())
    out = chain.route({"input": "search for python news"})
    assert out["destination"] == "web_search"
    assert out["next_inputs"]["input"] == "search for python news"


def test_langchain_chain_select_with_context(make_router):
    chain = VerQenRouterChain(make_router(), destinations={"web_search": object()})
    result = chain.select({"input": "search for python news", "context": "user is a dev"})
    assert result.agent == "web_search"
    assert chain.invoke({"input": "search for python news"})["destination"] == "web_search"


def test_langchain_chain_missing_input_raises(make_router):
    with pytest.raises(ValueError, match="non-empty 'input'"):
        VerQenRouterChain(make_router()).route({})


def test_langchain_chain_non_dict_inputs_raises(make_router):
    with pytest.raises(ValueError, match="dict"):
        VerQenRouterChain(make_router()).route("not a dict")  # type: ignore[arg-type]


# ---------------------------------------------------------------- crewai


def test_crewai_assign_returns_mapped_agent(make_router):
    sentinel = object()
    crew = CrewAIRouter(make_router(), agents={"web_search": sentinel})
    result, agent = crew.assign("search the news")
    assert result.agent == "web_search"
    assert agent is sentinel


def test_crewai_assign_without_mapping_returns_none(make_router):
    crew = CrewAIRouter(make_router())
    result, agent = crew.assign("search the news")
    assert result.agent == "web_search"
    assert agent is None


def test_crewai_assign_all(make_router):
    crew = CrewAIRouter(make_router(), agents={"web_search": object()})
    pairs = crew.assign_all(["search one", "search two"], context="ctx")
    assert len(pairs) == 2
    assert all(result.agent == "web_search" for result, _ in pairs)


# ---------------------------------------------------------------- autogen


def test_autogen_handoff(make_router):
    sentinel = object()
    gen = AutoGenRouter(make_router(), agents={"web_search": sentinel})
    result, agent = gen.handoff("search the news")
    assert result.agent == "web_search"
    assert agent is sentinel


def test_autogen_handoff_unmapped_returns_none(make_router):
    gen = AutoGenRouter(make_router())
    _, agent = gen.handoff("search the news")
    assert agent is None


# ---------------------------------------------------------------- llamaindex


class _FakeIndexEngine:
    def __init__(self):
        self.queries: list[str] = []

    def query(self, question: str) -> str:
        self.queries.append(question)
        return f"answer about {question}"


def test_llamaindex_route_and_query(make_router):
    engine = _FakeIndexEngine()
    index = LlamaIndexRouter(make_router(), destinations={"web_search": engine})
    result, destination = index.route("find the python docs")
    assert result.agent == "web_search"
    assert destination is engine
    assert index.query("find the python docs") == "answer about find the python docs"
    assert engine.queries == ["find the python docs"]


def test_llamaindex_unmapped_destination_raises(make_router):
    index = LlamaIndexRouter(make_router())
    with pytest.raises(KeyError, match="no query engine"):
        index.query("find the python docs")


# ---------------------------------------------------------------- openai tools


def test_agent_tool_schemas():
    schemas = agent_tool_schemas(
        [AgentSpec("web_search", "Searches the web"), "database_query"]
    )
    assert [schema["function"]["name"] for schema in schemas] == [
        "web_search",
        "database_query",
    ]
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["description"] == "Searches the web"
    assert schemas[1]["function"]["description"] == "database_query"
    assert schemas[0]["function"]["parameters"]["type"] == "object"


def test_select_tool_returns_matched_schema(make_router):
    tools = agent_tool_schemas(AGENTS)
    result, tool = select_tool(make_router(), tools, "search the web")
    assert result.agent == "web_search"
    assert tool["function"]["name"] == "web_search"


def test_select_tool_without_matching_schema(make_router):
    tools = agent_tool_schemas(["other_agent", "another_one"])
    result, tool = select_tool(make_router(), tools, "anything")
    assert result.agent == "web_search"
    assert tool is None


def test_tool_call_task():
    assert tool_call_task("search") == "search"
    assert tool_call_task("search", {}) == "search"
    assert tool_call_task("search", {"q": "news"}) == 'search: {"q": "news"}'
    assert tool_call_task("search", '{"q": "news"}') == 'search: {"q": "news"}'


def test_route_completion_requires_user_text(make_router):
    with pytest.raises(ValueError, match="no user text"):
        route_completion(
            make_router(),
            [{"role": "system", "content": "hi"}],
            agent_tool_schemas(AGENTS),
            client=object(),
        )


def test_route_completion_without_matching_tool(make_router):
    with pytest.raises(ValueError, match="no tool belongs"):
        route_completion(
            make_router(),
            [{"role": "user", "content": "search the web"}],
            agent_tool_schemas(["other_agent"]),
            client=object(),
        )


# ------------------------------------------------- framework-gated behaviour


@pytest.mark.skipif(_HAS_LANGCHAIN, reason="langchain installed")
def test_as_runnable_hint_when_langchain_missing(make_router):
    with pytest.raises(ImportError, match="LangChain is required"):
        integrations.as_runnable(VerQenRouterChain(make_router()))


@pytest.mark.skipif(_HAS_CREWAI, reason="crewai installed")
def test_build_crewai_agents_hint_when_missing():
    with pytest.raises(ImportError, match="CrewAI is required"):
        integrations.build_crewai_agents(AGENTS)


@pytest.mark.skipif(_HAS_AUTOGEN, reason="autogen installed")
def test_build_autogen_agents_hint_when_missing():
    with pytest.raises(ImportError, match="AutoGen is required"):
        integrations.build_autogen_agents(AGENTS)


@pytest.mark.skipif(_HAS_LLAMAINDEX, reason="llama-index installed")
def test_as_index_tool_hint_when_missing(make_router):
    with pytest.raises(ImportError, match="LlamaIndex is required"):
        integrations.as_index_tool(make_router())


@pytest.mark.skipif(_HAS_OPENAI, reason="openai installed")
def test_route_completion_hint_when_openai_missing(make_router):
    with pytest.raises(ImportError, match="openai is required"):
        route_completion(
            make_router(),
            [{"role": "user", "content": "search the web"}],
            agent_tool_schemas(AGENTS),
        )


@pytest.mark.skipif(not _HAS_LANGCHAIN, reason="langchain not installed")
def test_as_runnable_wraps_chain(make_router):
    runnable = integrations.as_runnable(VerQenRouterChain(make_router()))
    out = runnable.invoke({"input": "search for python news"})
    assert out["destination"] == "web_search"


@pytest.mark.skipif(not _HAS_CREWAI, reason="crewai not installed")
def test_build_crewai_agents_when_installed():
    agents = integrations.build_crewai_agents(["web_search", "database_query"])
    assert [agent.role for agent in agents] == ["web_search", "database_query"]


@pytest.mark.skipif(not _HAS_AUTOGEN, reason="autogen not installed")
def test_build_autogen_agents_when_installed():
    agents = integrations.build_autogen_agents(["web_search", "database_query"])
    assert sorted(agents) == ["database_query", "web_search"]


@pytest.mark.skipif(not _HAS_LLAMAINDEX, reason="llama-index not installed")
def test_as_index_tool_when_installed(make_router):
    tool = integrations.as_index_tool(make_router())
    assert tool.name == "verqen_router"
