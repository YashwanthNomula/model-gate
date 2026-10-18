"""Baseline regression comparison.

A deployment gate should catch slow decay, not just threshold breaches: a
model can pass every absolute threshold while quietly getting worse than
the version it replaces. ``mgate check --baseline prior-report.json``
compares the current artifact's ``metrics.json`` against the metrics
snapshot embedded in a previous JSON report and fails when any metric
regresses beyond its configured tolerance.

Tolerances live in the gates config::

    "regression": {
        "default_tolerance": 0.02,
        "tolerances": {"accuracy": 0.01, "loss": 0.005}
    }

A metric with neither a per-metric tolerance nor a default is skipped
("no tolerance configured") rather than guessed at. Direction comes from
the metrics-gate specs, so ``lower_is_better`` metrics regress when they
*rise*. Anything malformed — unreadable baseline, non-numeric values, a
baseline metric missing from the current run — fails closed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from modelgate.gates import GateResult, _parse_metric_spec


class BaselineError(ValueError):
    """Fail-closed error for baseline / tolerance problems."""


def load_metrics_snapshot(artifact):
    """Read the artifact's metrics.json (via its manifest) as {name: value}."""
    try:
        with open(os.path.join(artifact, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        metrics_path = os.path.join(
            artifact, manifest.get("metrics", "metrics.json")
        )
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineError(f"could not load current metrics: {exc}")
    if not isinstance(metrics, dict):
        raise BaselineError("metrics.json must be a JSON object of metric -> value")
    return metrics


def load_baseline_report(path):
    """Extract the metrics snapshot from a prior ``mgate check --format json`` report."""
    try:
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
    except OSError as exc:
        raise BaselineError(f"could not read baseline report {path}: {exc}")
    except json.JSONDecodeError as exc:
        raise BaselineError(f"invalid JSON in baseline report {path}: {exc}")
    if not isinstance(report, dict):
        raise BaselineError(f"baseline report {path} is not a JSON object")
    metrics = report.get("metrics")
    if metrics is None:
        raise BaselineError(
            f"baseline report {path} has no 'metrics' snapshot — "
            "regenerate it with `mgate check --format json` (v1.1.0+)"
        )
    if not isinstance(metrics, dict):
        raise BaselineError(f"baseline report {path}: 'metrics' must be an object")
    return metrics


def parse_tolerances(cfg):
    """Validate the ``regression`` config block.

    Returns (default_tolerance_or_None, {metric: tolerance}). Raises
    BaselineError on a malformed block so a bad config fails closed.
    """
    block = cfg.get("regression", {}) or {}
    if not isinstance(block, dict):
        raise BaselineError("'regression' config block must be an object")
    default = block.get("default_tolerance")
    if default is not None:
        if isinstance(default, bool) or not isinstance(default, (int, float)):
            raise BaselineError("'regression.default_tolerance' must be a number")
        if default < 0:
            raise BaselineError("'regression.default_tolerance' must be >= 0")
        default = float(default)
    per_metric = block.get("tolerances", {}) or {}
    if not isinstance(per_metric, dict):
        raise BaselineError("'regression.tolerances' must be an object")
    out = {}
    for name, tol in per_metric.items():
        if isinstance(tol, bool) or not isinstance(tol, (int, float)):
            raise BaselineError(f"'regression.tolerances.{name}' must be a number")
        if tol < 0:
            raise BaselineError(f"'regression.tolerances.{name}' must be >= 0")
        out[name] = float(tol)
    return default, out


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass
class MetricComparison:
    name: str
    current: float | None
    baseline: float | None
    delta: float | None
    tolerance: float | None
    direction: str
    outcome: str  # "ok" | "regressed" | "skipped" | "error"
    reason: str

    def detail(self):
        if self.outcome == "error":
            return f"{self.name}: {self.reason}"
        if self.outcome == "skipped":
            return f"{self.name}: skipped ({self.reason})"
        arrow = "↓" if self.delta < 0 else ("↑" if self.delta > 0 else "→")
        direction_note = " (lower is better)" if self.direction == "lower_is_better" else ""
        line = (
            f"{self.name} {self.current:.4g} vs baseline {self.baseline:.4g} "
            f"(Δ {arrow}{abs(self.delta):.4g}, tol {self.tolerance:.4g}){direction_note}"
        )
        if self.outcome == "regressed":
            line += " REGRESSED"
        return line


def compare_metrics(current, baseline, directions, tolerances, default_tolerance):
    """Compare current metrics against a baseline snapshot.

    `directions` maps metric name -> "higher_is_better" | "lower_is_better"
    (metrics absent from it default to higher_is_better). Returns a list of
    MetricComparison, one per baseline metric plus any current-only metric
    (reported as skipped: no baseline to compare against).
    """
    comparisons = []
    for name in sorted(baseline):
        base_val = baseline[name]
        direction = directions.get(name, "higher_is_better")
        if name not in current:
            comparisons.append(MetricComparison(
                name, None, base_val if _is_number(base_val) else None, None,
                None, direction, "error", "missing in current run"))
            continue
        cur_val = current[name]
        if not _is_number(cur_val) or not _is_number(base_val):
            comparisons.append(MetricComparison(
                name, cur_val if _is_number(cur_val) else None,
                base_val if _is_number(base_val) else None, None, None,
                direction, "error", "non-numeric metric value"))
            continue
        tol = tolerances.get(name, default_tolerance)
        if tol is None:
            comparisons.append(MetricComparison(
                name, float(cur_val), float(base_val), float(cur_val) - float(base_val),
                None, direction, "skipped", "no tolerance configured"))
            continue
        delta = float(cur_val) - float(base_val)
        # 1e-12 epsilon: 0.90 - 0.92 is -0.020000000000000018 in floats,
        # which must not count as "beyond a 0.02 tolerance".
        if direction == "lower_is_better":
            regressed = delta > tol + 1e-12
        else:
            regressed = delta < -(tol + 1e-12)
        comparisons.append(MetricComparison(
            name, float(cur_val), float(base_val), delta, tol, direction,
            "regressed" if regressed else "ok", ""))
    for name in sorted(set(current) - set(baseline)):
        comparisons.append(MetricComparison(
            name, current[name] if _is_number(current[name]) else None, None, None,
            None, directions.get(name, "higher_is_better"), "skipped",
            "no baseline value (new metric)"))
    return comparisons


def gate_regression(cfg, artifact, data_path, baseline_path):
    """Run the baseline-regression gate against a prior JSON report."""
    import time

    t0 = time.perf_counter()
    try:
        current = load_metrics_snapshot(artifact)
        baseline = load_baseline_report(baseline_path)
        default_tol, tolerances = parse_tolerances(cfg)
        specs = {}
        try:
            specs = {
                name: _parse_metric_spec(name, spec)
                for name, spec in dict(cfg.get("metrics", {})).items()
            }
        except ValueError as exc:
            return GateResult(
                "regression", "fail", str(exc),
                (time.perf_counter() - t0) * 1000.0,
            )
        directions = {name: direction for name, (_, direction) in specs.items()}
        comparisons = compare_metrics(
            current, baseline, directions, tolerances, default_tol
        )
    except BaselineError as exc:
        return GateResult(
            "regression", "fail", str(exc), (time.perf_counter() - t0) * 1000.0
        )
    errors = [c for c in comparisons if c.outcome == "error"]
    regressed = [c for c in comparisons if c.outcome == "regressed"]
    detail = "; ".join(c.detail() for c in comparisons) or "no metrics to compare"
    if errors:
        return GateResult(
            "regression", "fail", detail, (time.perf_counter() - t0) * 1000.0
        )
    if regressed:
        return GateResult(
            "regression", "fail", detail, (time.perf_counter() - t0) * 1000.0
        )
    if all(c.outcome == "skipped" for c in comparisons):
        return GateResult(
            "regression", "warn", detail, (time.perf_counter() - t0) * 1000.0
        )
    return GateResult(
        "regression", "pass", detail, (time.perf_counter() - t0) * 1000.0
    )
