"""Run the real Banking77 (77-class) routing benchmark via HuggingFace datasets.

This script is NOT imported by the test suite. It documents how to reproduce
the Banking77 number from the README (92.6% at 77 options) on your own
hardware.

Banking77 (https://huggingface.co/datasets/banking77) contains 77
online-banking intents with about 1000 labeled utterances each. Routing
among 77 intent names is the same decision problem as routing among 77
agents, which is why the README cites it as the many-option routing result.

Usage::

    pip install datasets          # not a dependency of verqen-router
    export VERQEN_CHECKPOINT=/path/to/verqen-checkpoint
    python benchmarks/banking77.py --limit 500
    python benchmarks/banking77.py --limit 500 --checkpoint /path/to/checkpoint

Requires the ``verqen`` package and a checkpoint; prints a clear message and
exits 1 when either is missing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # allow running without pip install

from verqen_router.engine import load_engine  # noqa: E402

QUESTION = "Which banking intent does this utterance belong to?"


def load_banking77(limit: int) -> tuple[list[str], list[dict]]:
    """Return ``(intent_names, rows)`` from the Banking77 test split.

    Exits with an install hint when the ``datasets`` package is missing, and
    raises ``SystemExit`` if the label vocabulary cannot be read.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "error: the 'datasets' package is required for the Banking77 benchmark.\n"
            "Install it with: pip install datasets\n"
            "(The bundled routing_dataset.jsonl is the offline regression suite;\n"
            " run benchmarks/run_benchmark.py instead.)",
            file=sys.stderr,
        )
        raise SystemExit(1)

    dataset = load_dataset("banking77", split="test")
    features = dataset.features.get("label")
    names = list(getattr(features, "names", []) or [])
    if not names:
        print(
            "error: could not read the Banking77 label names from the dataset features.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    rows = [
        {"text": row["text"], "expected": names[row["label"]]}
        for row in dataset
    ]
    if limit > 0:
        rows = rows[:limit]
    return names, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=None, help="VerQen checkpoint path")
    parser.add_argument("--limit", type=int, default=500, help="rows to evaluate (0 = all)")
    args = parser.parse_args(argv)

    try:
        engine = load_engine(args.checkpoint)
    except (ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    intents, rows = load_banking77(args.limit)
    print(f"evaluating {len(rows)} utterances over {len(intents)} intents")

    latencies: list[float] = []
    correct = 0
    started = time.time()
    for index, row in enumerate(rows, start=1):
        start = time.perf_counter()
        decision = engine.decide(row["text"], QUESTION, intents, question_type="choice")
        latencies.append((time.perf_counter() - start) * 1000)
        if decision.get("selected") == row["expected"]:
            correct += 1
        if index % 100 == 0 or index == len(rows):
            print(f"  {index}/{len(rows)}", flush=True)

    latencies.sort()
    accuracy = correct / max(1, len(rows))
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else 0.0
    print()
    print(f"accuracy     {accuracy:.1%} ({correct}/{len(rows)})")
    print(f"latency p50  {p50:.2f} ms  p95 {p95:.2f} ms")
    print(f"wall time    {time.time() - started:.1f} s")
    print("reference    README table: VerQen 92.6%, Jev 87.0%, Laya 35.8%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
