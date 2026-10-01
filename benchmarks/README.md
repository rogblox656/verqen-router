# Benchmark suite

Runs the bundled `routing_dataset.jsonl` through a live `verqen-router`.

```bash
export VERQEN_CHECKPOINT=/path/to/verqen-checkpoint
python benchmarks/run_benchmark.py
python benchmarks/run_benchmark.py --limit 20
python benchmarks/run_benchmark.py --fail-under 0.9   # CI gate, exit 3 on failure
```

## Dataset

Offline regression suite: 70 rows across 10 agents (`code_executor`,
`web_search`, `database_query`, `email_sender`, `calendar_api`,
`document_reader`, `translator`, `calculator`, `file_writer`,
`image_generator`). Each row:

```json
{"id": "route-001", "task": "...", "context": "...", "expected_agent": "code_executor"}
```

`context` may be an empty string. The agent list passed to the router is
derived from the dataset itself, so you can point `--dataset` at your own
JSONL with a different agent vocabulary.

## Metrics

* `accuracy`: predicted agent == expected agent
* `top_set_coverage`: expected agent inside the conformal set (the certified
  95% coverage claim, checked empirically)
* `fallback_rate`: routes that resolved without the model
* `latency_ms.p50 / p95 / max`: one `decide` call per row
* `per_agent_accuracy`: per expected agent
* `errors`: ids of missed rows

## Full public benchmarks (Banking77, CLINC150)

The official numbers in the README (97.5% tool/agent domain, 96.0% CLINC150,
92.6% Banking77 at 77 options) come from the full VerQen evaluation suites,
not from this smaller shipped set. Reproducing them requires downloading the
public datasets:

```bash
pip install datasets
python benchmarks/banking77.py --limit 500   # 77-class Banking77 intent routing
```

`banking77.py` downloads the HuggingFace `banking77` dataset on first run and
exits with an install hint if `datasets` is missing. The CLINC150 script has
the same shape (150 intents, `clinc150` on HuggingFace) and is not bundled to
keep the offline suite self-contained.

## Results

Reports are written to `results/benchmark-<timestamp>.json` (git-ignored).
Fields: `metrics.accuracy`, `top_set_coverage`, `fallback_rate`,
`mean_confidence`, `per_agent_accuracy`, `latency_ms.p50/p95/max`, `errors`,
plus every per-row record with its conformal set.
