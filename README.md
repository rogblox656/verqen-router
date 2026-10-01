# verqen-router

**3ms Multi-Agent Task Router: 97.5% accuracy, 318x cheaper than LLM routing.**

verqen-router replaces the expensive LLM step that decides *which agent or tool*
handles a task. It runs on your own machine as a single decision call: give it
your agent names, hand it a task plus optional context, and it returns the best
agent with a calibrated confidence and a certified conformal set, in about 3.5 ms.

```
Task + optional context
    |
    v
[VerQen Router] --- one decide() call over your agent names (2..64)
    |   - Which agent or tool should handle this task?   [choice]
    |   - probabilities + certified prediction set
    |
    +--> agent          -> dispatch to that agent/tool
    +--> {web_search, database_query}
                         -> ambiguous: run both, or escalate
    +--> fallback_agent -> engine unavailable or raised
```

## Why verqen-router

- **Every multi-agent system has an expensive LLM routing step.** LangChain
  router chains, CrewAI delegation, AutoGen group chat and OpenAI tool choice
  all burn an LLM call just to pick the next agent. That call costs ~450 ms and
  $14.64 per million routes with an API model.
- verqen-router answers the same question in **3.51 ms P50 for $0.046 per
  million routes** on a local GPU: **318x cheaper than Jev-1.13** and 133x
  cheaper than Laya, with *higher* accuracy than either.
- It scales to many agents: **92.6% accuracy over 77 options** (Banking77),
  so a registry of dozens of tools is no problem.
- Every route ships a **calibrated confidence** and a **conformal prediction
  set** certified at 95% coverage, so uncertain routes can run both agents
  instead of guessing.
- Zero hard dependencies: the core package is pure Python. The VerQen engine
  loads lazily, frameworks import lazily, and tests run without any of them.

## Benchmarks

| Metric | VerQen-T2 | Laya | Jev-1.13 |
|--------|-----------|------|----------|
| Tool/Agent Domain Accuracy | **97.5%** | 21.6% | 65.3% |
| Intent Classification (CLINC) | **96.0%** | ~78% | ~92% |
| Banking77 (77 options) | **92.6%** | 35.8% | 87.0% |
| Routing Latency P50 | **3.51 ms** | 26.85 ms | ~450 ms |
| Cost per 1M Routes | **$0.046** | $6.10 | $14.64 |

318x cheaper per route than Jev-1.13 (and 133x cheaper than Laya), while
winning every accuracy row.

Numbers from the VerQen evaluation suite (`jev_parity`), model card and
Banking77 split, live benchmark 2026-09-30. Latency is 3.51 ms with
`fast=True` compiled kernels; eager mode adds Python/dispatch overhead
(~68 ms on an RTX 5060 Ti) with identical accuracy. Methodology in the
[VerQen repository](https://github.com/ZorQelis-AI/verqen).

## Install

```bash
pip install verqen-router
# with the VerQen engine (local GPU inference):
pip install "verqen-router[gpu]"
pip install git+https://github.com/ZorQelis-AI/verqen.git
```

Set your checkpoint (once per shell, or in `.env`):

```bash
export VERQEN_CHECKPOINT=/path/to/verqen-checkpoint
```

## Quickstart

```python
from verqen_router import Router

router = Router(
    agents=["web_search", "database_query", "email_sender", "calendar_api"],
    fallback_agent="web_search",
)

result = router.route(
    task="Find the top customers by revenue",
    context="analytics dashboard",
)

print(result.agent)          # database_query
print(result.confidence)     # calibrated confidence, e.g. 0.94
print(result.conformal_set)  # certified set, e.g. ["database_query"]
print(result.is_ambiguous)   # True when the set names more than one agent
print(result.latency_ms)     # e.g. 3.51 (compiled) / 68 (eager)
```

Batch routing against shared context:

```python
results = router.route_multi(
    ["Send the standup notes", "Book a room for Friday"],
    context="team workspace, 12 people",
)
[result.agent for result in results]
```

Agents can be plain strings or `AgentSpec(name, description, keywords)` when
you want richer metadata for the registry and keyword fallback.

## Conformal routing sets

The engine returns a **conformal prediction set** alongside the top pick: at
the model's certified **95% coverage** level, the true agent is inside that
set. When the router is honestly torn between agents, you get the whole set:

```python
result = router.route("Look up today's weather and also compare flight prices")

result.conformal_set   # e.g. ["web_search", "database_query"]
result.is_ambiguous    # True (len(conformal_set) > 1)

if result.is_ambiguous:
    # certified at 95% coverage: run both agents, or escalate to a human
    ...
```

`conformal_set` always contains the selected agent, so `result.agent` is
never outside the certified set.

## State cache

The VerQen engine can encode a state once and answer many questions against
that single encoding. `StateCache` wraps this with an LRU cache, so N routing
questions over the same context cost 1 encode plus N cheap decisions:

```python
from verqen_router.cache import StateCache

cache = StateCache(router.engine, max_entries=128)
state = "User is debugging a Python service on Windows."

for task in tasks:
    decision = cache.decide(
        state,
        f"Which agent or tool should handle this task: {task}?",
        router.agent_names,
    )
    print(decision["selected"])

cache.stats()
# {'entries': 1, 'hits': 3, 'encodes': 1, 'avg_reuse_ms': 0.0197, ...}
```

Engines without `encode_state` / `decide_on_state` fall back to plain
`decide` calls automatically; results are identical either way.

## Integrations

All integrations lazy-import their framework, so none of them is required to
install or test verqen-router.

### LangChain

```python
from verqen_router import Router
from verqen_router.integrations import VerQenRouterChain

chain = VerQenRouterChain(router, destinations={"web_search": search_chain})
out = chain.route({"input": "search for python news"})
out["destination"]        # "web_search"
out["next_inputs"]        # the original inputs, LangChain RouterChain shape
# RunnableLambda wrapper: from verqen_router.integrations import as_runnable
```

### CrewAI

```python
from verqen_router.integrations import CrewAIRouter, build_crewai_agents

agents = build_crewai_agents(router.agents)      # needs pip install verqen-router[crewai]
crew_router = CrewAIRouter(router, dict(zip([a.name for a in router.agents], agents)))
result, crew_agent = crew_router.assign("Find the top customers by revenue")
```

### AutoGen

```python
from verqen_router.integrations import AutoGenRouter, build_autogen_agents

agents = build_autogen_agents(router.agents)     # needs pip install verqen-router[autogen]
result, agent = AutoGenRouter(router, agents).handoff("Run the load test script")
```

### OpenAI tools

```python
from verqen_router.integrations import agent_tool_schemas, select_tool

tools = agent_tool_schemas(router.agents)        # plain dicts, no openai import
result, tool = select_tool(router, tools, "search the news")
# route_completion(router, messages, tools) forces the chosen tool on an
# OpenAI call (needs pip install verqen-router[openai] and an API key).
```

### LlamaIndex

```python
from verqen_router.integrations import LlamaIndexRouter

index_router = LlamaIndexRouter(router, destinations={"document_reader": doc_engine})
answer = index_router.query("What does the contract say about termination?")
```

## Fallback behavior

Two layers keep routing alive when the model is not available:

1. **`fallback_agent=`** on `Router`: if the engine is missing (no
   `VERQEN_CHECKPOINT`) or raises during `route`, you get a result with
   `used_fallback=True` and `agent=fallback_agent` instead of an exception.
   Without `fallback_agent`, engine errors propagate to the caller.
2. **`AgentRegistry.quick_route(task, context)`**: keyword scoring over agent
   names and registered keywords, zero model calls. Tied scores all land in
   `conformal_set`; no signal at all gives `confidence=0.0` with the whole
   registry as the set.

```python
from verqen_router.registry import AgentRegistry

registry = AgentRegistry(["web_search", "database_query"])
registry.register("email_sender", description="Sends email", keywords=["email", "send"])

result = registry.quick_route("send an email to the team")
result.agent           # "email_sender"
result.used_fallback   # True (keyword path never touches the model)
```

## Benchmarks (this repo)

```bash
python benchmarks/run_benchmark.py --checkpoint "$VERQEN_CHECKPOINT"
python benchmarks/run_benchmark.py --fail-under 0.9   # CI gate, exit 3 on failure
```

Runs the bundled 70-row, 10-agent dataset and writes accuracy, top-set
coverage, fallback rate and latency percentiles to `benchmarks/results/`.
See `benchmarks/README.md` for the full Banking77 / CLINC150 reproduction
notes.

## Testing

```bash
pip install -e ".[dev]"
python -m pytest -q        # fast suite, no GPU/checkpoint needed
python -m pytest -q -m live  # live suite (needs VERQEN_CHECKPOINT)
```

Run pytest from the project directory so `python -m pytest` puts the project
root on `sys.path`. The fast suite uses an injected `FakeEngine`; no GPU,
torch, checkpoint or agent framework is required.

## Project layout

```
verqen_router/
  core.py            # Router, RouteResult, AgentSpec
  registry.py        # AgentRegistry + keyword quick_route fallback
  conformal.py       # prediction_set / is_ambiguous / confidence helpers
  cache.py           # StateCache (encode once, reuse per query)
  engine.py          # lazy VerQen engine loader
  integrations/      # LangChain, CrewAI, AutoGen, OpenAI tools, LlamaIndex
tests/               # 100+ tests, no GPU required
benchmarks/          # dataset + runner + Banking77 script
examples/            # runnable examples (need VERQEN_CHECKPOINT)
```

## License

Apache-2.0. Copyright ZorQelis AI. VerQen model: [github.com/ZorQelis-AI/verqen](https://github.com/ZorQelis-AI/verqen).
