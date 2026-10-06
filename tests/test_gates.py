"""Tests for model-gate. Zero dependencies besides pytest."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modelgate import demo
from modelgate.cli import main
from modelgate.gates import (
    GATE_DESCRIPTIONS,
    exit_code,
    gate_contract,
    gate_files,
    gate_latency,
    gate_metrics,
    gate_schema,
    gate_size,
    run_gates,
)
from modelgate.report import render_dict, render_html, render_json, summary_line


@pytest.fixture()
def passing(tmp_path):
    """A scaffolded demo that passes every gate."""
    out = str(tmp_path)
    artifact, sample, config = demo.scaffold(out)
    with open(config, encoding="utf-8") as f:
        cfg = json.load(f)
    return {"out": out, "artifact": artifact, "sample": sample, "cfg": cfg}


@pytest.fixture()
def broken(tmp_path):
    """A scaffolded demo that fails the schema + metrics gates."""
    out = str(tmp_path / "broken")
    os.makedirs(out, exist_ok=True)
    artifact, sample, config = demo.scaffold(out, broken=True)
    with open(config, encoding="utf-8") as f:
        cfg = json.load(f)
    return {"out": out, "artifact": artifact, "sample": sample, "cfg": cfg}


def _write_sample(path, columns, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")


# ---- files gate ------------------------------------------------------------

def test_files_gate_pass(passing):
    r = gate_files(passing["cfg"], passing["artifact"], passing["sample"])
    assert r.status == "pass"
    assert "5 required files" in r.detail


def test_files_gate_fail_missing(passing):
    os.remove(os.path.join(passing["artifact"], "model.bin"))
    r = gate_files(passing["cfg"], passing["artifact"], passing["sample"])
    assert r.status == "fail"
    assert "model.bin" in r.detail


# ---- schema gate -----------------------------------------------------------

def test_schema_gate_pass(passing):
    r = gate_schema(passing["cfg"], passing["artifact"], passing["sample"])
    assert r.status == "pass"
    assert "200 rows" in r.detail


def test_schema_gate_missing_column(passing):
    _write_sample(
        passing["sample"],
        ["amount", "merchant", "hour"],  # device dropped
        [["12.50", "grocery", "10"]] * 5,
    )
    r = gate_schema(passing["cfg"], passing["artifact"], passing["sample"])
    assert r.status == "fail"
    assert "device" in r.detail


def test_schema_gate_type_coercion_failure(passing):
    _write_sample(
        passing["sample"],
        ["amount", "merchant", "hour", "device"],
        [["12.50", "grocery", "not-an-hour", "mobile"]] * 5,
    )
    r = gate_schema(passing["cfg"], passing["artifact"], passing["sample"])
    assert r.status == "fail"
    assert "type coercion" in r.detail


def test_schema_gate_missing_rate_failure(passing):
    rows = [["12.50", "grocery", "10", "mobile"]] * 90 + [["", "grocery", "10", "mobile"]] * 10
    _write_sample(passing["sample"], ["amount", "merchant", "hour", "device"], rows)
    cfg = dict(passing["cfg"])
    cfg["schema"] = {"max_missing_rate": 0.0}
    r = gate_schema(cfg, passing["artifact"], passing["sample"])
    assert r.status == "fail"
    assert "missing rate" in r.detail


def test_schema_gate_extra_columns_noted(passing):
    rows = [["12.50", "grocery", "10", "mobile", "x1"]] * 5
    _write_sample(passing["sample"], ["amount", "merchant", "hour", "device", "extra"], rows)
    r = gate_schema(passing["cfg"], passing["artifact"], passing["sample"])
    assert r.status == "pass"
    assert "extra columns" in r.detail


def test_schema_gate_no_data_rows(passing):
    _write_sample(passing["sample"], ["amount", "merchant", "hour", "device"], [])
    r = gate_schema(passing["cfg"], passing["artifact"], passing["sample"])
    assert r.status == "fail"


# ---- metrics gate ----------------------------------------------------------

def test_metrics_gate_pass(passing):
    r = gate_metrics(passing["cfg"], passing["artifact"], passing["sample"])
    assert r.status == "pass"
    assert "accuracy" in r.detail


def test_metrics_gate_missing_metric(passing):
    with open(os.path.join(passing["artifact"], "metrics.json"), "w") as f:
        json.dump({"accuracy": 0.93}, f)
    r = gate_metrics(passing["cfg"], passing["artifact"], passing["sample"])
    assert r.status == "fail"
    assert "missing metrics" in r.detail


def test_metrics_gate_below_threshold(passing):
    with open(os.path.join(passing["artifact"], "metrics.json"), "w") as f:
        json.dump({"accuracy": 0.93, "f1": 0.50, "auc": 0.95}, f)
    r = gate_metrics(passing["cfg"], passing["artifact"], passing["sample"])
    assert r.status == "fail"
    assert "f1" in r.detail


# ---- latency gate ----------------------------------------------------------

def test_latency_gate_pass_and_percentiles(passing):
    r = gate_latency(passing["cfg"], passing["artifact"], passing["sample"])
    assert r.status == "pass"
    assert "p50" in r.detail and "p95" in r.detail and "p99" in r.detail


def test_latency_gate_fail_over_budget(passing):
    cfg = dict(passing["cfg"])
    cfg["latency"] = dict(cfg["latency"], p95_ms=0.0001)
    r = gate_latency(cfg, passing["artifact"], passing["sample"])
    assert r.status == "fail"
    assert "budget" in r.detail


def test_latency_gate_predict_error(passing):
    with open(os.path.join(passing["artifact"], "model.py"), "a") as f:
        f.write("\nraise RuntimeError('boom')\n")
    r = gate_latency(passing["cfg"], passing["artifact"], passing["sample"])
    assert r.status == "fail"


# ---- size gate -------------------------------------------------------------

def test_size_gate_pass(passing):
    r = gate_size(passing["cfg"], passing["artifact"], passing["sample"])
    assert r.status == "pass"
    assert "model.bin" in r.detail


def test_size_gate_oversize(passing):
    cfg = dict(passing["cfg"])
    cfg["size"] = {"file": "model.bin", "max_bytes": 10}
    r = gate_size(cfg, passing["artifact"], passing["sample"])
    assert r.status == "fail"


def test_size_gate_missing_file(passing):
    cfg = dict(passing["cfg"])
    cfg["size"] = {"file": "nope.bin", "max_bytes": 100}
    r = gate_size(cfg, passing["artifact"], passing["sample"])
    assert r.status == "fail"


# ---- contract gate ---------------------------------------------------------

def test_contract_gate_pass(passing):
    r = gate_contract(passing["cfg"], passing["artifact"], passing["sample"])
    assert r.status == "pass"


def test_contract_gate_unknown_label(passing):
    with open(os.path.join(passing["artifact"], "model.py"), "a") as f:
        f.write("\n\ndef predict(records):\n    return ['maybe'] * len(records)\n")
    r = gate_contract(passing["cfg"], passing["artifact"], passing["sample"])
    assert r.status == "fail"
    assert "maybe" in r.detail


def test_contract_gate_length_mismatch(passing):
    with open(os.path.join(passing["artifact"], "model.py"), "a") as f:
        f.write("\n\ndef predict(records):\n    return ['legit']\n")
    r = gate_contract(passing["cfg"], passing["artifact"], passing["sample"])
    assert r.status == "fail"
    assert "1 outputs" in r.detail


# ---- end-to-end: check command ---------------------------------------------

def test_check_exit_0_passing(passing):
    code = main(["check", passing["artifact"], "--config",
                 os.path.join(passing["out"], "gates.json"),
                 "--data", passing["sample"]])
    assert code == 0


def test_check_exit_1_broken(broken):
    code = main(["check", broken["artifact"], "--config",
                 os.path.join(broken["out"], "gates.json"),
                 "--data", broken["sample"]])
    assert code == 1


def test_broken_demo_fails_exactly_two_gates(broken):
    results = run_gates(broken["cfg"], broken["artifact"], broken["sample"])
    failed = {r.name for r in results if r.status == "fail"}
    assert failed == {"schema", "metrics"}
    assert summary_line(results).startswith("4/6 passed")


def test_check_missing_config(passing):
    code = main(["check", passing["artifact"], "--config", "nope.json",
                 "--data", passing["sample"]])
    assert code == 2


def test_init_scaffolds_passing_demo(tmp_path, capsys):
    code = main(["init", "--out", str(tmp_path)])
    assert code == 0
    out = str(tmp_path)
    code = main(["check", os.path.join(out, "demo-model"),
                 "--config", os.path.join(out, "gates.json"),
                 "--data", os.path.join(out, "sample.csv")])
    assert code == 0
    assert "6/6 passed" in capsys.readouterr().out


def test_init_broken_scaffolds_failing_demo(tmp_path):
    code = main(["init", "--broken", "--out", str(tmp_path)])
    assert code == 0
    out = str(tmp_path)
    code = main(["check", os.path.join(out, "demo-model"),
                 "--config", os.path.join(out, "gates.json"),
                 "--data", os.path.join(out, "sample.csv")])
    assert code == 1


def test_gates_command_lists_all(capsys):
    assert main(["gates"]) == 0
    out = capsys.readouterr().out
    for name in GATE_DESCRIPTIONS:
        assert name in out


# ---- reports ---------------------------------------------------------------

def test_json_report_shape(passing):
    results = run_gates(passing["cfg"], passing["artifact"], passing["sample"])
    payload = json.loads(render_json(results, "demo-model"))
    assert payload["artifact"] == "demo-model"
    assert payload["summary"]["total"] == 6
    assert payload["summary"]["pass"] == 6
    gate = payload["gates"][0]
    assert set(gate) == {"name", "status", "detail", "duration_ms"}


def test_render_dict_has_all_status_keys(passing):
    results = run_gates(passing["cfg"], passing["artifact"], passing["sample"])
    d = render_dict(results, "x")
    assert set(d["summary"]) >= {"total", "pass", "fail", "warn", "skip"}


def test_html_report_standalone(passing):
    results = run_gates(passing["cfg"], passing["artifact"], passing["sample"])
    page = render_html(results, "demo-model")
    assert "<!DOCTYPE html>" in page
    assert "6/6 passed" in page
    assert 'src="' not in page and "href=" not in page  # no external assets


def test_exit_code_allows_warnings(passing):
    from modelgate.gates import GateResult
    results = [GateResult("x", "warn", "d", 1.0), GateResult("y", "pass", "d", 1.0)]
    assert exit_code(results) == 0
    assert exit_code([GateResult("x", "fail", "d", 1.0)]) == 1
