"""Run the bundled routing dataset through verqen-router and record metrics.

Usage::

    python benchmarks/run_benchmark.py --checkpoint "$VERQEN_CHECKPOINT"
    python benchmarks/run_benchmark.py --limit 20 --fail-under 0.9

Loads ``routing_dataset.jsonl`` (70 task rows across 10 agents), routes every
task with a live router (one ``decide`` call per row), and writes a JSON
report to ``benchmarks/results/`` with:

* accuracy: predicted agent == expected agent
* top_set_coverage: expected agent inside the conformal set
* fallback_rate: share of routes that fell back without the model
* latency p50 / p95

The bundled dataset is a regression suite. The official VerQen benchmark
numbers in the README come from the full evaluation suites (tool/agent
domain, CLINC150, Banking77); see ``benchmarks/README.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "routing_dataset.jsonl"
RESULTS_DIR = HERE / "results"

sys.path.insert(0, str(HERE.parent))  # allow running without pip install

from verqen_router import Router  # noqa: E402


def load_dataset(path: Path = DEFAULT_DATASET) -> list[dict]:
    """Load and validate the JSONL dataset (id, task, context, expected_agent)."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = {"id", "task", "context", "expected_agent"} - row.keys()
            if missing:
                raise ValueError(f"{path.name}:{lineno}: missing fields {sorted(missing)}")
            if not str(row["task"]).strip():
                raise ValueError(f"{path.name}:{lineno}: empty task")
            if not str(row["expected_agent"]).strip():
                raise ValueError(f"{path.name}:{lineno}: empty expected_agent")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path.name}: dataset is empty")
    return rows


def route_row(router: Router, row: dict) -> dict:
    """Route one dataset row and time the full call."""
    start = time.perf_counter()
    result = router.route(row["task"], context=row.get("context", ""))
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "id": row["id"],
        "expected": row["expected_agent"],
        "predicted": result.agent,
        "correct": result.agent == row["expected_agent"],
        "covered": row["expected_agent"] in result.conformal_set,
        "used_fallback": result.used_fallback,
        "confidence": round(result.confidence, 4),
        "conformal_set": list(result.conformal_set),
        "latency_ms": round(elapsed_ms, 3),
    }


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100); 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), -(-int(p * len(ordered)) // 100)))
    return ordered[rank - 1]


def summarize(records: list[dict]) -> dict:
    """Aggregate per-record results into a metrics report."""
    total = len(records)
    if total == 0:
        raise ValueError("no records to summarize")
    correct = sum(1 for record in records if record["correct"])
    covered = sum(1 for record in records if record["covered"])
    fallbacks = sum(1 for record in records if record["used_fallback"])

    by_agent: dict[str, list[dict]] = {}
    for record in records:
        by_agent.setdefault(record["expected"], []).append(record)

    latencies = [record["latency_ms"] for record in records]
    return {
        "total": total,
        "accuracy": round(correct / total, 4),
        "top_set_coverage": round(covered / total, 4),
        "fallback_rate": round(fallbacks / total, 4),
        "mean_confidence": round(
            sum(record["confidence"] for record in records) / total, 4
        ),
        "per_agent_accuracy": {
            name: round(sum(1 for record in rows if record["correct"]) / len(rows), 4)
            for name, rows in sorted(by_agent.items())
        },
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 3),
            "p95": round(percentile(latencies, 95), 3),
            "max": round(max(latencies), 3),
        },
        "errors": [record["id"] for record in records if not record["correct"]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=None, help="VerQen checkpoint path")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=0, help="only run the first N rows")
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="exit 3 when overall accuracy is below this value",
    )
    args = parser.parse_args(argv)

    rows = load_dataset(args.dataset)
    agents = list(dict.fromkeys(row["expected_agent"] for row in rows))
    if args.limit > 0:
        rows = rows[: args.limit]

    try:
        router = Router(agents=agents, checkpoint=args.checkpoint)
        router.engine  # resolve the lazy engine now so a missing checkpoint fails fast
    except (ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    started = time.time()
    records = []
    for index, row in enumerate(rows, start=1):
        records.append(route_row(router, row))
        if index % 20 == 0 or index == len(rows):
            print(f"  {index}/{len(rows)}", flush=True)

    report = {
        "dataset": str(args.dataset.name),
        "agents": agents,
        "rows": len(records),
        "wall_seconds": round(time.time() - started, 2),
        "metrics": summarize(records),
        "records": records,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"benchmark-{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    metrics = report["metrics"]
    print()
    print(f"dataset      {report['dataset']} ({report['rows']} rows, {len(agents)} agents)")
    print(f"accuracy     {metrics['accuracy']:.1%}")
    print(f"top-set      {metrics['top_set_coverage']:.1%} coverage")
    print(f"fallback     {metrics['fallback_rate']:.1%}")
    print(
        f"latency p50  {metrics['latency_ms']['p50']} ms  "
        f"p95 {metrics['latency_ms']['p95']} ms"
    )
    print(f"report       {out_path}")
    if metrics["errors"]:
        print(f"misses ({len(metrics['errors'])}): {', '.join(metrics['errors'][:20])}")

    if args.fail_under is not None and metrics["accuracy"] < args.fail_under:
        print(
            f"FAIL: accuracy {metrics['accuracy']:.3f} < {args.fail_under:.3f}",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
