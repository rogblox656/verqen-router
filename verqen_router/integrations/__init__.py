"""Framework integrations for verqen-router.

Each module adapts a :class:`~verqen_router.core.Router` to one agent
framework. Framework-specific imports stay inside the functions that need
them, so importing this package never requires LangChain, CrewAI, AutoGen,
LlamaIndex or OpenAI.
"""

from __future__ import annotations

__all__ = [
    "VerQenRouterChain",
    "as_runnable",
    "CrewAIRouter",
    "build_crewai_agents",
    "AutoGenRouter",
    "build_autogen_agents",
    "agent_tool_schemas",
    "select_tool",
    "tool_call_task",
    "route_completion",
    "LlamaIndexRouter",
    "as_index_tool",
]


def __getattr__(name: str):
    # Lazy re-exports keep optional frameworks optional.
    if name == "VerQenRouterChain":
        from verqen_router.integrations.langchain import VerQenRouterChain

        return VerQenRouterChain
    if name == "as_runnable":
        from verqen_router.integrations.langchain import as_runnable

        return as_runnable
    if name == "CrewAIRouter":
        from verqen_router.integrations.crewai import CrewAIRouter

        return CrewAIRouter
    if name == "build_crewai_agents":
        from verqen_router.integrations.crewai import build_crewai_agents

        return build_crewai_agents
    if name == "AutoGenRouter":
        from verqen_router.integrations.autogen import AutoGenRouter

        return AutoGenRouter
    if name == "build_autogen_agents":
        from verqen_router.integrations.autogen import build_autogen_agents

        return build_autogen_agents
    if name == "agent_tool_schemas":
        from verqen_router.integrations.openai_tools import agent_tool_schemas

        return agent_tool_schemas
    if name == "select_tool":
        from verqen_router.integrations.openai_tools import select_tool

        return select_tool
    if name == "tool_call_task":
        from verqen_router.integrations.openai_tools import tool_call_task

        return tool_call_task
    if name == "route_completion":
        from verqen_router.integrations.openai_tools import route_completion

        return route_completion
    if name == "LlamaIndexRouter":
        from verqen_router.integrations.llamaindex import LlamaIndexRouter

        return LlamaIndexRouter
    if name == "as_index_tool":
        from verqen_router.integrations.llamaindex import as_index_tool

        return as_index_tool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
