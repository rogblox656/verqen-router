"""OpenAI tool routing: let VerQen choose the tool instead of the LLM."""

from __future__ import annotations

import json
from typing import Any, Iterable

from verqen_router.core import AgentSpec, RouteResult, Router

__all__ = ["agent_tool_schemas", "select_tool", "tool_call_task", "route_completion"]


def agent_tool_schemas(specs: Iterable[AgentSpec | str]) -> list[dict]:
    """OpenAI function-tool schemas for agents (name + description).

    Pure dict building, no ``openai`` import: register the returned schemas
    as your assistant's ``tools``, then let :func:`select_tool` pick one.
    """
    schemas = []
    for spec in specs:
        normalized = AgentSpec(spec, spec) if isinstance(spec, str) else spec
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": normalized.name,
                    "description": normalized.description
                    or f"Handle the task with {normalized.name}",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            }
        )
    return schemas


def tool_call_task(name: str, arguments: dict[str, Any] | str = "") -> str:
    """Task text for an OpenAI tool call: the tool name plus its arguments."""
    if isinstance(arguments, str):
        payload = arguments.strip()
    else:
        payload = json.dumps(arguments or {}, sort_keys=True)
    if payload and payload != "{}":
        return f"{name}: {payload}"
    return name


def select_tool(
    router: Router,
    tools: list[dict],
    task: str,
    context: str = "",
) -> tuple[RouteResult, dict | None]:
    """Route *task* among OpenAI *tools* and return ``(result, chosen_schema)``.

    The router's agent names are expected to match the tools'
    ``function.name`` values; the matched schema comes back as the second
    element (``None`` when no tool belongs to the routed agent).
    """
    result = router.route(task=task, context=context)
    for tool in tools:
        if _tool_name(tool) == result.agent:
            return result, tool
    return result, None


def route_completion(
    router: Router,
    messages: list[dict],
    tools: list[dict],
    *,
    client: Any | None = None,
    model: str = "gpt-4o-mini",
) -> tuple[Any, RouteResult]:
    """Route the last user message to one tool, then call OpenAI with that tool forced.

    Returns ``(completion, route_result)``. When no ``client`` is passed the
    ``openai`` package is imported here; raises ``ImportError`` with an
    install hint if it is missing.
    """
    task = _last_user_text(messages)
    result, schema = select_tool(router, tools, task)
    if schema is None:
        raise ValueError(f"no tool belongs to routed agent {result.agent!r}")
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is required for route_completion() when no client is passed. "
                "Install it with: pip install verqen-router[openai]"
            ) from exc
        client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": schema["function"]["name"]}},
    )
    return completion, result


def _tool_name(tool: dict) -> str:
    function = tool.get("function") or {}
    return str(function.get("name") or tool.get("name") or "")


def _last_user_text(messages: Iterable[dict]) -> str:
    for message in reversed(list(messages)):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
    raise ValueError("messages contain no user text to route")
