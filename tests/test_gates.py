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
    parse_severities,
    resolve_overrides,
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
    assert set(gate) == {"name", "status", "detail", "duration_ms", "severity"}
    assert gate["severity"] == "error"  # default severity is recorded, not implied


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


# ---- gate overrides (exception workflow) -----------------------------------

def test_override_waives_failing_gate(broken):
    cfg = dict(broken["cfg"])
    cfg["overrides"] = {"metrics": "f1 dip approved by on-call, retrain scheduled"}
    results = run_gates(cfg, broken["artifact"], broken["sample"],
                        overrides=resolve_overrides(cfg))
    by_name = {r.name: r for r in results}
    assert by_name["metrics"].status == "skip"
    assert "f1 dip approved by on-call" in by_name["metrics"].detail
    assert by_name["schema"].status == "fail"  # not waived, still fails
    assert exit_code(results) == 1
    assert summary_line(results) == "4/6 passed (1 failed, 1 skipped)"


def test_override_all_failures_exits_0(broken):
    cfg = dict(broken["cfg"])
    cfg["overrides"] = {"schema": "sample predates country rollout",
                        "metrics": "approved in incident #4821"}
    results = run_gates(cfg, broken["artifact"], broken["sample"],
                        overrides=resolve_overrides(cfg))
    assert all(r.status != "fail" for r in results)
    assert exit_code(results) == 0


def test_override_gate_is_not_executed(passing):
    """A waived gate must not run at all — no duration, no side effects."""
    with open(os.path.join(passing["artifact"], "metrics.json"), "w") as f:
        f.write("not valid json {{{")  # would hard-fail gate_metrics if executed
    cfg = dict(passing["cfg"])
    results = run_gates(cfg, passing["artifact"], passing["sample"],
                        overrides=resolve_overrides(cfg, [("metrics", "waived for test")]))
    by_name = {r.name: r for r in results}
    assert by_name["metrics"].status == "skip"
    assert by_name["metrics"].duration_ms == 0.0


def test_override_unknown_gate_fails_closed(passing):
    cfg = dict(passing["cfg"])
    cfg["overrides"] = {"not-a-gate": "typo"}
    try:
        resolve_overrides(cfg)
    except ValueError as exc:
        assert "not-a-gate" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown gate override")


def test_override_empty_justification_rejected(passing):
    cfg = dict(passing["cfg"])
    cfg["overrides"] = {"metrics": "   "}
    try:
        resolve_overrides(cfg)
    except ValueError as exc:
        assert "justification" in str(exc)
    else:
        raise AssertionError("expected ValueError for empty justification")


def test_cli_override_flag_waives_gate(broken, capsys):
    code = main(["check", broken["artifact"], "--config",
                 os.path.join(broken["out"], "gates.json"),
                 "--data", broken["sample"],
                 "--override", "metrics=f1 dip approved, retrain scheduled"])
    assert code == 1  # schema still fails
    out = capsys.readouterr().out
    assert "waived by override" in out
    assert "f1 dip approved" in out


def test_cli_override_malformed_pair_rejected(broken):
    code = main(["check", broken["artifact"], "--config",
                 os.path.join(broken["out"], "gates.json"),
                 "--data", broken["sample"],
                 "--override", "metrics"])  # missing =reason
    assert code == 2


def test_cli_override_unknown_gate_rejected(broken):
    code = main(["check", broken["artifact"], "--config",
                 os.path.join(broken["out"], "gates.json"),
                 "--data", broken["sample"],
                 "--override", "nope=bad gate name"])
    assert code == 2


def test_json_report_records_override_reason(broken):
    cfg = dict(broken["cfg"])
    cfg["overrides"] = {"metrics": "approved waiver"}
    results = run_gates(cfg, broken["artifact"], broken["sample"],
                        overrides=resolve_overrides(cfg))
    payload = json.loads(render_json(results, "demo-model"))
    metric_gate = next(g for g in payload["gates"] if g["name"] == "metrics")
    assert metric_gate["status"] == "skip"
    assert "approved waiver" in metric_gate["detail"]
    assert payload["summary"]["skip"] == 1


# ---- baseline regression comparison --------------------------------------

from modelgate.baseline import (
    BaselineError,
    compare_metrics,
    gate_regression,
    load_baseline_report,
    parse_tolerances,
)


def _baseline_report(tmp_path, metrics, name="baseline.json"):
    path = os.path.join(str(tmp_path), name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"artifact": "demo-model", "metrics": metrics}, f)
    return path


def _regression_cfg(passing, regression, metrics_values=None):
    cfg = dict(passing["cfg"])
    cfg["regression"] = regression
    if metrics_values is not None:
        with open(os.path.join(passing["artifact"], "metrics.json"), "w") as f:
            json.dump(metrics_values, f)
    return cfg


def test_json_report_embeds_metrics_snapshot(passing):
    results = run_gates(passing["cfg"], passing["artifact"], passing["sample"])
    payload = json.loads(render_json(
        results, "demo-model",
        metrics={"accuracy": 0.93, "f1": 0.87, "auc": 0.95}))
    assert payload["metrics"] == {"accuracy": 0.93, "f1": 0.87, "auc": 0.95}


def test_json_report_without_snapshot_has_no_metrics_key(passing):
    """Old callers get the exact previous shape — the key is additive only."""
    results = run_gates(passing["cfg"], passing["artifact"], passing["sample"])
    payload = json.loads(render_json(results, "demo-model"))
    assert "metrics" not in payload


def test_regression_gate_pass_within_tolerance(passing, tmp_path):
    baseline = _baseline_report(
        tmp_path, {"accuracy": 0.94, "f1": 0.88, "auc": 0.95})
    cfg = _regression_cfg(passing, {"default_tolerance": 0.02})
    r = gate_regression(cfg, passing["artifact"], passing["sample"], baseline)
    assert r.status == "pass"
    assert "accuracy 0.93 vs baseline 0.94" in r.detail


def test_regression_gate_fail_beyond_tolerance(passing, tmp_path):
    baseline = _baseline_report(
        tmp_path, {"accuracy": 0.94, "f1": 0.95, "auc": 0.95})
    cfg = _regression_cfg(passing, {"default_tolerance": 0.02})
    r = gate_regression(cfg, passing["artifact"], passing["sample"], baseline)
    assert r.status == "fail"
    assert "f1" in r.detail and "REGRESSED" in r.detail
    assert "accuracy" in r.detail  # non-regressed metrics still reported


def test_regression_gate_per_metric_tolerance(passing, tmp_path):
    """A per-metric tolerance overrides the default for that metric only."""
    baseline = _baseline_report(
        tmp_path, {"accuracy": 0.94, "f1": 0.95, "auc": 0.95})
    cfg = _regression_cfg(
        passing,
        {"default_tolerance": 0.02, "tolerances": {"f1": 0.10}})
    r = gate_regression(cfg, passing["artifact"], passing["sample"], baseline)
    assert r.status == "pass"


def test_regression_gate_lower_is_better_direction(passing, tmp_path):
    cfg = _regression_cfg(
        passing,
        {"default_tolerance": 0.01},
        metrics_values={"loss": 0.40},
    )
    cfg["metrics"] = {"loss": {"threshold": 0.5, "direction": "lower_is_better"}}
    # loss rose 0.32 -> 0.40: +0.08 beyond tol 0.01 -> regressed
    baseline = _baseline_report(tmp_path, {"loss": 0.32})
    r = gate_regression(cfg, passing["artifact"], passing["sample"], baseline)
    assert r.status == "fail"
    assert "REGRESSED" in r.detail and "lower is better" in r.detail
    # loss fell instead: an improvement, never a regression
    baseline2 = _baseline_report(tmp_path, {"loss": 0.50}, name="baseline2.json")
    r2 = gate_regression(cfg, passing["artifact"], passing["sample"], baseline2)
    assert r2.status == "pass"


def test_regression_gate_missing_baseline_file_fails_closed(passing):
    cfg = _regression_cfg(passing, {"default_tolerance": 0.02})
    r = gate_regression(cfg, passing["artifact"], passing["sample"],
                        "/nonexistent/baseline.json")
    assert r.status == "fail"
    assert "could not read baseline" in r.detail


def test_regression_gate_baseline_without_snapshot_fails_closed(passing, tmp_path):
    path = os.path.join(str(tmp_path), "old.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"artifact": "demo-model", "gates": []}, f)
    cfg = _regression_cfg(passing, {"default_tolerance": 0.02})
    r = gate_regression(cfg, passing["artifact"], passing["sample"], path)
    assert r.status == "fail"
    assert "no 'metrics' snapshot" in r.detail


def test_regression_gate_missing_current_metric_fails_closed(passing, tmp_path):
    baseline = _baseline_report(
        tmp_path, {"accuracy": 0.94, "f1": 0.88, "auc": 0.95, "recall": 0.90})
    cfg = _regression_cfg(passing, {"default_tolerance": 0.02})
    r = gate_regression(cfg, passing["artifact"], passing["sample"], baseline)
    assert r.status == "fail"
    assert "recall" in r.detail and "missing in current run" in r.detail


def test_regression_gate_new_metric_skipped_not_failed(passing, tmp_path):
    """A metric with no baseline value is new work, not a regression."""
    baseline = _baseline_report(tmp_path, {"accuracy": 0.94})
    cfg = _regression_cfg(passing, {"default_tolerance": 0.02})
    r = gate_regression(cfg, passing["artifact"], passing["sample"], baseline)
    assert r.status == "pass"
    assert "f1: skipped (no baseline value" in r.detail


def test_regression_gate_no_tolerance_skips_with_warn(passing, tmp_path):
    baseline = _baseline_report(tmp_path, {"accuracy": 0.94, "f1": 0.88})
    cfg = _regression_cfg(passing, {})  # no default, no per-metric
    r = gate_regression(cfg, passing["artifact"], passing["sample"], baseline)
    assert r.status == "warn"  # nothing comparable: visible, but exit 0
    assert "no tolerance configured" in r.detail
    assert exit_code([r]) == 0


def test_regression_gate_negative_tolerance_fails_closed(passing, tmp_path):
    baseline = _baseline_report(tmp_path, {"accuracy": 0.94})
    cfg = _regression_cfg(passing, {"tolerances": {"accuracy": -0.01}})
    r = gate_regression(cfg, passing["artifact"], passing["sample"], baseline)
    assert r.status == "fail"
    assert ">= 0" in r.detail


def test_regression_gate_non_numeric_value_fails_closed(passing, tmp_path):
    baseline = _baseline_report(tmp_path, {"accuracy": "high"})
    cfg = _regression_cfg(passing, {"default_tolerance": 0.02})
    r = gate_regression(cfg, passing["artifact"], passing["sample"], baseline)
    assert r.status == "fail"
    assert "non-numeric" in r.detail


def test_parse_tolerances_rejects_bad_types():
    with pytest.raises(BaselineError):
        parse_tolerances({"regression": {"default_tolerance": True}})
    with pytest.raises(BaselineError):
        parse_tolerances({"regression": {"tolerances": {"a": "0.01"}}})


def test_compare_metrics_boundary_is_not_regression():
    """A drop exactly equal to the tolerance is allowed."""
    out = compare_metrics({"a": 0.90}, {"a": 0.92}, {}, {}, 0.02)
    assert out[0].outcome == "ok"
    out = compare_metrics({"a": 0.89}, {"a": 0.92}, {}, {}, 0.02)
    assert out[0].outcome == "regressed"


def _write_cfg(tmp_path, cfg, name="cfg.json"):
    path = os.path.join(str(tmp_path), name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return path


def test_check_with_baseline_end_to_end(passing, tmp_path):
    baseline = _baseline_report(tmp_path, {"accuracy": 0.94, "f1": 0.88, "auc": 0.95})
    cfg = _regression_cfg(passing, {"default_tolerance": 0.02})
    config_path = _write_cfg(tmp_path, cfg, "reg_gates.json")
    code = main(["check", passing["artifact"], "--config", config_path,
                 "--data", passing["sample"], "--baseline", baseline])
    assert code == 0  # gates pass and no metric regressed beyond tolerance


def test_check_with_baseline_regression_exits_1(passing, tmp_path, capsys):
    baseline = _baseline_report(tmp_path, {"accuracy": 0.94, "f1": 0.99, "auc": 0.95})
    cfg = _regression_cfg(passing, {"default_tolerance": 0.02})
    config_path = _write_cfg(tmp_path, cfg, "reg_gates.json")
    code = main(["check", passing["artifact"], "--config", config_path,
                 "--data", passing["sample"], "--baseline", baseline])
    assert code == 1
    out = capsys.readouterr().out
    assert "regression" in out and "REGRESSED" in out


def test_check_baseline_json_snapshot_roundtrip(passing, tmp_path):
    """A JSON report from this version works as the next run's baseline."""
    cfg = _regression_cfg(passing, {"default_tolerance": 0.02})
    config_path = _write_cfg(tmp_path, cfg, "reg_gates.json")
    snap = os.path.join(str(tmp_path), "snap.json")
    code = main(["check", passing["artifact"], "--config", config_path,
                 "--data", passing["sample"], "--format", "json",
                 "--output", snap])
    assert code == 0
    with open(snap, encoding="utf-8") as f:
        assert json.load(f)["metrics"]["accuracy"] == 0.93
    code = main(["check", passing["artifact"], "--config", config_path,
                 "--data", passing["sample"], "--baseline", snap])
    assert code == 0  # identical metrics: zero delta, no regression


def test_check_only_regression_needs_baseline(passing, tmp_path):
    cfg = _regression_cfg(passing, {"default_tolerance": 0.02})
    config_path = _write_cfg(tmp_path, cfg, "reg_gates.json")
    code = main(["check", passing["artifact"], "--config", config_path,
                 "--data", passing["sample"], "--only", "regression"])
    assert code == 1  # fail-closed: regression gate without a baseline file


def _metric_cfg_with(passing, metrics_cfg, metrics_values):
    cfg = dict(passing["cfg"])
    cfg["metrics"] = metrics_cfg
    with open(os.path.join(passing["artifact"], "metrics.json"), "w") as f:
        json.dump(metrics_values, f)
    return cfg


def test_metrics_gate_lower_is_better_pass(passing):
    cfg = _metric_cfg_with(
        passing,
        {"loss": {"threshold": 0.5, "direction": "lower_is_better"}},
        {"loss": 0.32},
    )
    r = gate_metrics(cfg, passing["artifact"], passing["sample"])
    assert r.status == "pass"
    assert "loss 0.32 ≤ 0.5 (lower is better)" in r.detail


def test_metrics_gate_lower_is_better_boundary_passes(passing):
    """Value exactly at the threshold passes (≤ semantics)."""
    cfg = _metric_cfg_with(
        passing,
        {"loss": {"threshold": 0.5, "direction": "lower_is_better"}},
        {"loss": 0.5},
    )
    r = gate_metrics(cfg, passing["artifact"], passing["sample"])
    assert r.status == "pass"


def test_metrics_gate_lower_is_better_fail(passing):
    cfg = _metric_cfg_with(
        passing,
        {"loss": {"threshold": 0.5, "direction": "lower_is_better"},
         "p95_latency_ms": {"threshold": 100.0, "direction": "lower_is_better"}},
        {"loss": 0.61, "p95_latency_ms": 80.0},
    )
    r = gate_metrics(cfg, passing["artifact"], passing["sample"])
    assert r.status == "fail"
    assert "above threshold" in r.detail
    assert "loss" in r.detail
    assert "lower is better" in r.detail
    assert "p95_latency_ms" not in r.detail  # under budget, no breach


def test_metrics_gate_direction_defaults_to_higher_is_better(passing):
    """Object form without `direction` behaves exactly like the bare number form."""
    cfg = _metric_cfg_with(
        passing,
        {"accuracy": {"threshold": 0.9}},
        {"accuracy": 0.93},
    )
    r = gate_metrics(cfg, passing["artifact"], passing["sample"])
    assert r.status == "pass"
    assert "accuracy 0.93 ≥ 0.9" in r.detail
    assert "lower is better" not in r.detail


def test_metrics_gate_explicit_higher_is_better(passing):
    cfg = _metric_cfg_with(
        passing,
        {"accuracy": {"threshold": 0.9, "direction": "higher_is_better"}},
        {"accuracy": 0.85},
    )
    r = gate_metrics(cfg, passing["artifact"], passing["sample"])
    assert r.status == "fail"
    assert "below threshold" in r.detail


def test_metrics_gate_mixed_directions(passing):
    cfg = _metric_cfg_with(
        passing,
        {"accuracy": 0.9,
         "loss": {"threshold": 0.5, "direction": "lower_is_better"}},
        {"accuracy": 0.93, "loss": 0.32},
    )
    r = gate_metrics(cfg, passing["artifact"], passing["sample"])
    assert r.status == "pass"
    assert "accuracy 0.93 ≥ 0.9" in r.detail
    assert "loss 0.32 ≤ 0.5 (lower is better)" in r.detail


def test_metrics_gate_invalid_direction_fails_closed(passing):
    """An unknown direction string fails the gate instead of passing silently."""
    cfg = _metric_cfg_with(
        passing,
        {"loss": {"threshold": 0.5, "direction": "smaller_is_better"}},
        {"loss": 0.32},
    )
    r = gate_metrics(cfg, passing["artifact"], passing["sample"])
    assert r.status == "fail"
    assert "invalid direction" in r.detail
    assert "smaller_is_better" in r.detail
    assert exit_code([r]) == 1


def test_metrics_gate_object_form_missing_threshold_fails_closed(passing):
    cfg = _metric_cfg_with(
        passing,
        {"loss": {"direction": "lower_is_better"}},
        {"loss": 0.32},
    )
    r = gate_metrics(cfg, passing["artifact"], passing["sample"])
    assert r.status == "fail"
    assert "threshold" in r.detail


def test_metrics_gate_bare_number_still_default_higher(passing):
    """Legacy bare-number config keeps ≥ semantics, byte-identical detail."""
    r = gate_metrics(passing["cfg"], passing["artifact"], passing["sample"])
    assert r.status == "pass"
    assert r.detail == "accuracy 0.93 ≥ 0.9, f1 0.87 ≥ 0.8, auc 0.95 ≥ 0.9"


def test_metric_direction_surfaced_in_all_reports(passing):
    """The direction tag must be auditable in terminal, JSON, and HTML reports."""
    cfg = _metric_cfg_with(
        passing,
        {"accuracy": 0.9,
         "loss": {"threshold": 0.5, "direction": "lower_is_better"}},
        {"accuracy": 0.93, "loss": 0.32},
    )
    results = run_gates(cfg, passing["artifact"], passing["sample"])
    metric = next(r for r in results if r.name == "metrics")
    assert "(lower is better)" in metric.detail
    # terminal
    from modelgate.report import render_text
    assert "(lower is better)" in render_text(results, "demo-model")
    # JSON
    payload = json.loads(render_json(results, "demo-model"))
    json_gate = next(g for g in payload["gates"] if g["name"] == "metrics")
    assert "(lower is better)" in json_gate["detail"]
    # HTML
    assert "(lower is better)" in render_html(results, "demo-model")


def test_check_command_with_directions_end_to_end(passing):
    cfg = dict(passing["cfg"])
    cfg["metrics"] = {
        "accuracy": 0.9,
        "loss": {"threshold": 0.5, "direction": "lower_is_better"},
    }
    with open(os.path.join(passing["artifact"], "metrics.json"), "w") as f:
        json.dump({"accuracy": 0.93, "f1": 0.87, "auc": 0.95, "loss": 0.61}, f)
    config_path = os.path.join(passing["out"], "dir_gates.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    code = main(["check", passing["artifact"], "--config", config_path,
                 "--data", passing["sample"]])
    assert code == 1  # loss over threshold fails the run


# ---- warning-severity gates ------------------------------------------------

def _warn_cfg(passing, metrics_cfg=None, metrics_values=None, severity=None):
    """Passing-demo config with a metrics gate that fails + a severity block."""
    cfg = dict(passing["cfg"])
    cfg["metrics"] = metrics_cfg or {"accuracy": 0.9, "f1": 0.8, "auc": 0.9}
    with open(os.path.join(passing["artifact"], "metrics.json"), "w") as f:
        json.dump(metrics_values or
                  {"accuracy": 0.93, "f1": 0.50, "auc": 0.95}, f)  # f1 breaches
    cfg["severity"] = severity if severity is not None else {"metrics": "warn"}
    return cfg


def test_warn_severity_failure_records_fail_but_exits_0(passing):
    """A warn-severity gate failure is recorded honestly but does not block."""
    cfg = _warn_cfg(passing)
    results = run_gates(cfg, passing["artifact"], passing["sample"])
    by_name = {r.name: r for r in results}
    assert by_name["metrics"].status == "fail"  # still FAIL in the record
    assert by_name["metrics"].severity == "warn"
    assert exit_code(results) == 0  # deployment proceeds
    from modelgate.report import summary_line
    assert summary_line(results) == "5/6 passed (1 warning)"


def test_default_severity_error_still_blocks(passing):
    """No severity block: identical failure keeps the old blocking behavior."""
    cfg = _warn_cfg(passing, severity={})
    results = run_gates(cfg, passing["artifact"], passing["sample"])
    by_name = {r.name: r for r in results}
    assert by_name["metrics"].severity == "error"
    assert exit_code(results) == 1
    from modelgate.report import summary_line
    assert summary_line(results) == "5/6 passed (1 failed)"


def test_mixed_severities_error_failure_blocks(passing):
    """An error-severity failure blocks even alongside a warn failure."""
    cfg = _warn_cfg(passing)
    cfg["size"] = {"file": "model.bin", "max_bytes": 10}  # size (error) fails too
    results = run_gates(cfg, passing["artifact"], passing["sample"])
    by_name = {r.name: r for r in results}
    assert by_name["size"].severity == "error"
    assert by_name["metrics"].severity == "warn"
    assert exit_code(results) == 1
    from modelgate.report import summary_line
    assert summary_line(results) == "4/6 passed (1 failed, 1 warning)"


def test_severity_unknown_string_fails_closed(passing):
    cfg = dict(passing["cfg"])
    cfg["severity"] = {"metrics": "critical"}
    with pytest.raises(ValueError, match="invalid severity"):
        run_gates(cfg, passing["artifact"], passing["sample"])


def test_severity_unknown_gate_fails_closed(passing):
    cfg = dict(passing["cfg"])
    cfg["severity"] = {"not-a-gate": "warn"}
    with pytest.raises(ValueError, match="unknown gate"):
        run_gates(cfg, passing["artifact"], passing["sample"])


def test_severity_block_must_be_object(passing):
    cfg = dict(passing["cfg"])
    cfg["severity"] = ["metrics"]
    with pytest.raises(ValueError, match="must be an object"):
        run_gates(cfg, passing["artifact"], passing["sample"])


def test_parse_severities_defaults_and_known_gates():
    assert parse_severities({}) == {}
    assert parse_severities({"severity": {"metrics": "warn"}}) == {"metrics": "warn"}
    # "regression" is not in run_gates' table but is a valid severity target
    assert parse_severities({"severity": {"regression": "warn"}}) == {
        "regression": "warn"}
    assert parse_severities({"severity": {"metrics": "error"}}) == {"metrics": "error"}


def test_check_warn_severity_exit_0_terminal_section(broken, tmp_path, capsys):
    """End-to-end: warn-severity failures exit 0 with a WARNINGS section."""
    cfg = dict(broken["cfg"])
    cfg["severity"] = {"schema": "warn", "metrics": "warn"}
    config_path = _write_cfg(tmp_path, cfg, "warn_gates.json")
    code = main(["check", broken["artifact"], "--config", config_path,
                 "--data", broken["sample"]])
    assert code == 0
    out = capsys.readouterr().out
    assert "4/6 passed (2 warning)" in out
    assert "WARNINGS" in out
    assert "severity: warn" in out


def test_check_strict_promotes_warn_to_error(broken, tmp_path):
    """--strict: every warn-severity failure blocks the deployment."""
    cfg = dict(broken["cfg"])
    cfg["severity"] = {"schema": "warn", "metrics": "warn"}
    config_path = _write_cfg(tmp_path, cfg, "warn_gates.json")
    code = main(["check", broken["artifact"], "--config", config_path,
                 "--data", broken["sample"], "--strict"])
    assert code == 1


def test_check_malformed_severity_exit_2(broken, tmp_path):
    cfg = dict(broken["cfg"])
    cfg["severity"] = {"metrics": "critical"}
    config_path = _write_cfg(tmp_path, cfg, "badsev.json")
    code = main(["check", broken["artifact"], "--config", config_path,
                 "--data", broken["sample"]])
    assert code == 2


def test_check_strict_still_rejects_malformed_severity(broken, tmp_path):
    """--strict promotes warn gates, but a bad severity block still fails closed."""
    cfg = dict(broken["cfg"])
    cfg["severity"] = {"metrics": "critical"}
    config_path = _write_cfg(tmp_path, cfg, "badsev.json")
    code = main(["check", broken["artifact"], "--config", config_path,
                 "--data", broken["sample"], "--strict"])
    assert code == 2


def test_json_report_warnings_list_and_severity_keys(passing):
    cfg = _warn_cfg(passing)
    results = run_gates(cfg, passing["artifact"], passing["sample"])
    payload = json.loads(render_json(results, "demo-model"))
    assert [g["name"] for g in payload["warnings"]] == ["metrics"]
    gate = next(g for g in payload["gates"] if g["name"] == "metrics")
    assert gate["status"] == "fail"
    assert gate["severity"] == "warn"
    assert payload["summary"]["fail"] == 1  # status counted honestly


def test_html_warn_banner(passing):
    cfg = _warn_cfg(passing)
    results = run_gates(cfg, passing["artifact"], passing["sample"])
    page = render_html(results, "demo-model")
    assert "model-gate: PASS (with warnings)" in page
    assert 'class="banner warn"' in page
    assert "[severity: warn]" in page


def test_html_fail_banner_unchanged(broken):
    """Error-severity failures keep the red FAIL banner."""
    results = run_gates(broken["cfg"], broken["artifact"], broken["sample"])
    page = render_html(results, "demo-model")
    assert "model-gate: FAIL" in page
    assert 'class="banner bad"' in page


def test_regression_gate_honours_warn_severity(passing, tmp_path):
    """A warn-severity regression failure records FAIL but exits 0."""
    baseline = _baseline_report(
        tmp_path, {"accuracy": 0.99, "f1": 0.88, "auc": 0.95})
    cfg = _regression_cfg(passing, {"default_tolerance": 0.02})
    cfg["severity"] = {"regression": "warn"}
    r = gate_regression(cfg, passing["artifact"], passing["sample"], baseline)
    assert r.status == "fail"
    assert "REGRESSED" in r.detail
    assert r.severity == "warn"
    assert exit_code([r]) == 0
