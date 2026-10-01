"""Benchmark harness tests: dataset integrity and metric math (no GPU needed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from verqen_router import Router

_BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "run_benchmark.py"
_spec = importlib.util.spec_from_file_location("run_benchmark", _BENCH)
run_benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_benchmark)


def test_dataset_loads_and_validates():
    rows = run_benchmark.load_dataset()
    assert 60 <= len(rows) <= 100
    assert all({"id", "task", "context", "expected_agent"} <= row.keys() for row in rows)
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)), "duplicate dataset ids"
    agents = {row["expected_agent"] for row in rows}
    assert len(agents) >= 8


def test_dataset_tasks_are_substantial():
    for row in run_benchmark.load_dataset():
        assert row["task"].strip()
        assert len(row["task"]) >= 10
        assert isinstance(row["context"], str)
        assert row["expected_agent"].strip()


def test_percentile():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert run_benchmark.percentile(values, 50) == 3.0
    assert run_benchmark.percentile(values, 95) == 5.0
    assert run_benchmark.percentile([], 50) == 0.0


def _record(record_id, expected, predicted, *, covered, fallback=False, latency=5.0):
    return {
        "id": record_id,
        "expected": expected,
        "predicted": predicted,
        "correct": predicted == expected,
        "covered": covered,
        "used_fallback": fallback,
        "confidence": 0.9,
        "conformal_set": [predicted],
        "latency_ms": latency,
    }


def test_summarize_perfect_predictions():
    records = [
        _record("a", "web_search", "web_search", covered=True, latency=10.0),
        _record("b", "email_sender", "email_sender", covered=True, latency=30.0),
    ]
    summary = run_benchmark.summarize(records)
    assert summary["accuracy"] == 1.0
    assert summary["top_set_coverage"] == 1.0
    assert summary["fallback_rate"] == 0.0
    assert summary["mean_confidence"] == 0.9
    # nearest-rank: rank = ceil(0.5 * 2) = 1 -> first value.
    assert summary["latency_ms"]["p50"] == 10.0
    assert summary["errors"] == []


def test_summarize_counts_misses_coverage_and_fallbacks():
    records = [
        _record("a", "web_search", "email_sender", covered=False, latency=5.0),
        _record("b", "web_search", "email_sender", covered=True, latency=5.0),
        _record("c", "email_sender", "email_sender", covered=True, fallback=True, latency=5.0),
    ]
    summary = run_benchmark.summarize(records)
    assert summary["accuracy"] == pytest.approx(1 / 3, abs=1e-4)
    assert summary["top_set_coverage"] == pytest.approx(2 / 3, abs=1e-4)
    assert summary["fallback_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert summary["per_agent_accuracy"] == {"email_sender": 1.0, "web_search": 0.0}
    assert summary["errors"] == ["a", "b"]


def test_summarize_empty_raises():
    with pytest.raises(ValueError):
        run_benchmark.summarize([])


def test_main_fails_cleanly_without_checkpoint(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("VERQEN_CHECKPOINT", raising=False)
    dataset = tmp_path / "tiny.jsonl"
    dataset.write_text(
        '{"id": "x1", "task": "search the web for news", "context": "", '
        '"expected_agent": "web_search"}\n'
        '{"id": "x2", "task": "send an email to the team", "context": "", '
        '"expected_agent": "email_sender"}\n',
        encoding="utf-8",
    )
    assert run_benchmark.main(["--dataset", str(dataset)]) == 1
    assert "VERQEN_CHECKPOINT" in capsys.readouterr().err


def test_fail_under_gate(monkeypatch, capsys, tmp_path, fake_engine_cls):
    """With a canned engine only some rows match: accuracy gates the exit code."""
    dataset = tmp_path / "gate.jsonl"
    dataset.write_text(
        '{"id": "x1", "task": "search the web for news", "context": "", '
        '"expected_agent": "web_search"}\n'
        '{"id": "x2", "task": "send an email to the team", "context": "", '
        '"expected_agent": "email_sender"}\n',
        encoding="utf-8",
    )

    def _factory(**router_kwargs):
        return Router(engine=fake_engine_cls(selected="web_search"), **router_kwargs)

    monkeypatch.setattr(run_benchmark, "Router", _factory)
    monkeypatch.setattr(run_benchmark, "RESULTS_DIR", tmp_path)

    # Canned engine gets 1 of 2 rows right: accuracy is 0.5.
    assert run_benchmark.main(["--dataset", str(dataset), "--fail-under", "0.6"]) == 3
    assert "FAIL: accuracy" in capsys.readouterr().err
    assert run_benchmark.main(["--dataset", str(dataset), "--fail-under", "0.4"]) == 0
    capsys.readouterr()
    reports = list(tmp_path.glob("benchmark-*.json"))
    assert reports, "report JSON was not written"
