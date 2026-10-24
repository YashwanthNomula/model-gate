"""Gate implementations.

Each gate inspects the model artifact (and the validation sample where
relevant) and returns a GateResult with a name, a status
(pass / fail / warn / skip), a human-readable detail, and a duration.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import time
from dataclasses import dataclass

PASS, FAIL, WARN, SKIP = "pass", "fail", "warn", "skip"

# Per-gate severity: "error" gates block the deployment when they fail;
# "warn" gates record the failure but let the deployment proceed (exit 0).
SEVERITY_ERROR, SEVERITY_WARN = "error", "warn"
VALID_SEVERITIES = frozenset({SEVERITY_ERROR, SEVERITY_WARN})

HIGHER_IS_BETTER = "higher_is_better"
LOWER_IS_BETTER = "lower_is_better"
VALID_DIRECTIONS = frozenset({HIGHER_IS_BETTER, LOWER_IS_BETTER})

GATE_DESCRIPTIONS = {
    "files": "Required artifact files exist",
    "schema": "Sample data matches the declared schema (fields, types, missing rate)",
    "metrics": "Recorded metrics meet configured thresholds",
    "latency": "predict() p95 latency stays under budget on repeated batches",
    "size": "Artifact file fits within the size limit",
    "contract": "predict() output length and labels satisfy the serving contract",
    "regression": "Metrics do not regress beyond tolerance vs a prior JSON report (needs --baseline)",
}

DEFAULT_FILES = ["manifest.json", "model.py", "schema.json", "metrics.json"]


@dataclass
class GateResult:
    name: str
    status: str
    detail: str
    duration_ms: float = 0.0
    severity: str = SEVERITY_ERROR
    """Gate severity: "error" blocks the deployment on failure,
    "warn" records the failure but leaves the verdict PASS (exit 0)."""

    def to_dict(self):
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "duration_ms": round(self.duration_ms, 2),
            "severity": self.severity,
        }


def _now():
    return time.perf_counter()


def _elapsed(t0):
    return (time.perf_counter() - t0) * 1000.0


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_manifest(artifact):
    return _load_json(os.path.join(artifact, "manifest.json"))


def _coerces(value, typ):
    """Check a raw CSV string coerces to the declared type."""
    v = value.strip()
    try:
        if typ == "int":
            int(v)
        elif typ == "float":
            float(v)
        elif typ == "bool":
            if v.lower() not in ("true", "false", "1", "0", "yes", "no"):
                return False
        elif typ == "string":
            pass
        else:
            return False
        return True
    except (ValueError, TypeError):
        return False


def _read_rows(data_path):
    with open(data_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return (reader.fieldnames or []), list(reader)


def _percentile(sorted_vals, pct):
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _fmt_bytes(n):
    for unit, size in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= size:
            return f"{n / size:.1f}{unit}"
    return f"{n}B"


def gate_files(cfg, artifact, data_path):
    t0 = _now()
    required = cfg.get("files", {}).get("required", DEFAULT_FILES)
    missing = [f for f in required if not os.path.isfile(os.path.join(artifact, f))]
    if missing:
        return GateResult(
            "files", FAIL, f"missing: {', '.join(missing)}", _elapsed(t0)
        )
    return GateResult(
        "files",
        PASS,
        f"{len(required)} required files present",
        _elapsed(t0),
    )


def gate_schema(cfg, artifact, data_path):
    t0 = _now()
    scfg = cfg.get("schema", {})
    max_missing = float(scfg.get("max_missing_rate", 0.01))
    try:
        manifest = _read_manifest(artifact)
        schema_path = os.path.join(artifact, manifest.get("schema", "schema.json"))
        schema = _load_json(schema_path)
        columns, rows = _read_rows(data_path)
    except Exception as exc:  # unreadable artifact/sample is a hard fail
        return GateResult("schema", FAIL, f"could not load schema/data: {exc}", _elapsed(t0))
    if not rows:
        return GateResult("schema", FAIL, "sample CSV has no data rows", _elapsed(t0))

    problems = []
    missing_cols = [c for c in schema if c not in columns]
    if missing_cols:
        problems.append(f"missing columns: {', '.join(missing_cols)}")

    type_errors = 0
    missing_vals = 0
    checkable = [c for c in schema if c in columns]
    for row in rows:
        for col in checkable:
            v = (row.get(col) or "").strip()
            if v == "":
                missing_vals += 1
            elif not _coerces(v, schema[col]):
                type_errors += 1
    denom = max(len(rows) * len(schema), 1)
    miss_rate = missing_vals / denom
    if type_errors:
        problems.append(f"{type_errors} value(s) fail type coercion")
    if miss_rate > max_missing:
        problems.append(
            f"missing rate {miss_rate:.3%} > max {max_missing:.3%}"
        )

    extras = [c for c in columns if c not in schema]
    note = f"; extra columns noted (ignored): {', '.join(extras)}" if extras else ""
    if problems:
        return GateResult(
            "schema", FAIL, "; ".join(problems) + note, _elapsed(t0)
        )
    return GateResult(
        "schema",
        PASS,
        f"{len(rows)} rows x {len(schema)} fields ok, missing rate {miss_rate:.3%}{note}",
        _elapsed(t0),
    )


def _parse_metric_spec(name, spec):
    """Parse one metric gate entry into (threshold, direction).

    Accepts a bare number (backwards-compatible, higher_is_better) or
    {"threshold": <number>, "direction": "higher_is_better"|"lower_is_better"}.
    Raises ValueError on anything else — callers fail the gate closed.
    """
    if isinstance(spec, bool):
        raise ValueError(f"metric '{name}': threshold must be a number or a "
                         "{'threshold', 'direction'} object")
    if isinstance(spec, (int, float)):
        return float(spec), HIGHER_IS_BETTER
    if isinstance(spec, dict):
        if "threshold" not in spec:
            raise ValueError(f"metric '{name}': object form needs a 'threshold'")
        threshold = spec["threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(f"metric '{name}': threshold must be a number")
        direction = spec.get("direction", HIGHER_IS_BETTER)
        if direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"metric '{name}': invalid direction '{direction}' "
                "(expected 'higher_is_better' or 'lower_is_better')"
            )
        return float(threshold), direction
    raise ValueError(f"metric '{name}': threshold must be a number or a "
                     "{'threshold', 'direction'} object")


def gate_metrics(cfg, artifact, data_path):
    t0 = _now()
    mcfg = dict(cfg.get("metrics", {}))
    try:
        manifest = _read_manifest(artifact)
        metrics_path = os.path.join(
            artifact, manifest.get("metrics", "metrics.json")
        )
        metrics = _load_json(metrics_path)
    except Exception as exc:
        return GateResult("metrics", FAIL, f"could not load metrics: {exc}", _elapsed(t0))
    if not mcfg:
        return GateResult("metrics", SKIP, "no metric thresholds configured", _elapsed(t0))

    # Parse specs first: a malformed entry fails the gate closed, before any
    # comparison runs.
    try:
        specs = {k: _parse_metric_spec(k, spec) for k, spec in mcfg.items()}
    except ValueError as exc:
        return GateResult("metrics", FAIL, str(exc), _elapsed(t0))

    missing = [k for k in specs if k not in metrics]
    if missing:
        return GateResult(
            "metrics", FAIL, f"missing metrics: {', '.join(missing)}", _elapsed(t0)
        )
    problems = []
    for k, (threshold, direction) in specs.items():
        value = metrics[k]
        if direction == LOWER_IS_BETTER:
            if value > threshold:
                problems.append(
                    f"above threshold: {k} {value:.4g} > {threshold:.4g} (lower is better)"
                )
        elif value < threshold:
            problems.append(f"below threshold: {k} {value:.4g} < {threshold:.4g}")
    if problems:
        return GateResult(
            "metrics", FAIL, "; ".join(problems), _elapsed(t0)
        )
    parts = []
    for k, (threshold, direction) in specs.items():
        value = metrics[k]
        if direction == LOWER_IS_BETTER:
            parts.append(f"{k} {value:.4g} ≤ {threshold:.4g} (lower is better)")
        else:
            parts.append(f"{k} {value:.4g} ≥ {threshold:.4g}")
    return GateResult("metrics", PASS, ", ".join(parts), _elapsed(t0))


def _load_model(artifact):
    manifest = _read_manifest(artifact)
    entry = manifest.get("entrypoint", "model.py")
    path = os.path.join(artifact, entry)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"entrypoint not found: {entry}")
    spec = importlib.util.spec_from_file_location("gate_demo_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "predict", None)):
        raise TypeError(f"{entry} must expose predict(records)")
    return module


def gate_latency(cfg, artifact, data_path):
    t0 = _now()
    lcfg = cfg.get("latency", {})
    n_calls = int(lcfg.get("n_calls", 10))
    batch = int(lcfg.get("batch", 100))
    budget = float(lcfg.get("p95_ms", 500.0))
    try:
        model = _load_model(artifact)
        _, rows = _read_rows(data_path)
        records = rows[:batch] or [{}]
        times_ms = []
        for _ in range(n_calls):
            s = _now()
            model.predict(records)
            times_ms.append((_now() - s) * 1000.0)
    except Exception as exc:
        return GateResult("latency", FAIL, f"prediction error: {exc}", _elapsed(t0))

    ordered = sorted(times_ms)
    p50 = _percentile(ordered, 50)
    p95 = _percentile(ordered, 95)
    p99 = _percentile(ordered, 99)
    detail = (
        f"p50 {p50:.2f}ms, p95 {p95:.2f}ms, p99 {p99:.2f}ms "
        f"over {n_calls}x{len(records)} predict calls"
    )
    if p95 > budget:
        return GateResult(
            "latency", FAIL, f"{detail}; budget p95 ≤ {budget:.2f}ms", _elapsed(t0)
        )
    if p95 > 0.8 * budget:
        return GateResult(
            "latency", WARN, f"{detail}; p95 within 20% of budget {budget:.2f}ms", _elapsed(t0)
        )
    return GateResult("latency", PASS, detail, _elapsed(t0))


def gate_size(cfg, artifact, data_path):
    t0 = _now()
    scfg = cfg.get("size", {})
    fname = scfg.get("file", "model.bin")
    max_bytes = int(scfg.get("max_bytes", 10 * 1024 * 1024))
    path = os.path.join(artifact, fname)
    if not os.path.isfile(path):
        return GateResult("size", FAIL, f"{fname} not found in artifact", _elapsed(t0))
    size = os.path.getsize(path)
    detail = f"{fname} {_fmt_bytes(size)} (limit {_fmt_bytes(max_bytes)})"
    if size > max_bytes:
        return GateResult("size", FAIL, detail, _elapsed(t0))
    return GateResult("size", PASS, detail, _elapsed(t0))


def gate_contract(cfg, artifact, data_path):
    t0 = _now()
    ccfg = cfg.get("contract", {})
    n_rows = int(ccfg.get("n_rows", 50))
    try:
        manifest = _read_manifest(artifact)
        labels = manifest.get("labels", [])
        model = _load_model(artifact)
        _, rows = _read_rows(data_path)
        sample = rows[:n_rows]
        preds = model.predict(sample)
    except Exception as exc:
        return GateResult("contract", FAIL, f"prediction error: {exc}", _elapsed(t0))
    if len(preds) != len(sample):
        return GateResult(
            "contract",
            FAIL,
            f"predict returned {len(preds)} outputs for {len(sample)} inputs",
            _elapsed(t0),
        )
    bad = sorted({str(p) for p in preds if p not in labels})
    if bad:
        return GateResult(
            "contract",
            FAIL,
            f"unknown labels: {', '.join(bad)} (allowed: {', '.join(labels)})",
            _elapsed(t0),
        )
    return GateResult(
        "contract",
        PASS,
        f"{len(sample)} rows in, {len(preds)} labels out, all ∈ {labels}",
        _elapsed(t0),
    )


_GATES = {
    "files": gate_files,
    "schema": gate_schema,
    "metrics": gate_metrics,
    "latency": gate_latency,
    "size": gate_size,
    "contract": gate_contract,
}


def parse_severities(cfg, known_gates=None):
    """Parse the per-gate ``severity`` config block.

    Returns {gate_name: "error"|"warn"}. Gates absent from the block keep
    the default severity "error", preserving current behavior.
    Raises ValueError on anything malformed — a severity naming an
    unknown gate, or an unknown severity string, fails closed: a
    silently-ignored severity would misreport the deployment verdict.
    """
    block = cfg.get("severity", {}) or {}
    if not isinstance(block, dict):
        raise ValueError(
            "'severity' config block must be an object mapping gate -> 'error'|'warn'"
        )
    known = set(known_gates) if known_gates is not None else set(_GATES) | {"regression"}
    out = {}
    for name, sev in block.items():
        if name not in known:
            raise ValueError(f"severity for unknown gate '{name}'")
        if not isinstance(sev, str) or sev not in VALID_SEVERITIES:
            raise ValueError(
                f"severity for gate '{name}': invalid severity '{sev}' "
                "(expected 'error' or 'warn')"
            )
        out[name] = sev
    return out


def run_gates(cfg, artifact, data_path, only=None, overrides=None, severities=None):
    """Run the configured gates (or the `only` subset) and return GateResults.

    Gates listed in `overrides` are waived: they are not executed and
    return a skip result carrying the justification string, so the
    waiver is recorded in every report format.

    `severities` is the {gate: "error"|"warn"} map from parse_severities
    (parsed from cfg when omitted); each result is stamped with its
    gate's severity so the exit code and reports can tell a blocking
    failure from a recorded warning.
    """
    wanted = list(only) if only is not None else list(cfg.get("gates", _GATES))
    overrides = overrides or {}
    sev = severities if severities is not None else parse_severities(cfg)
    results = []
    for name in wanted:
        gate_sev = sev.get(name, SEVERITY_ERROR)
        if name in overrides:
            results.append(
                GateResult(
                    name,
                    SKIP,
                    f"waived by override: {overrides[name]}",
                    0.0,
                    severity=gate_sev,
                )
            )
            continue
        fn = _GATES.get(name)
        if fn is None:
            results.append(
                GateResult(name, SKIP, f"unknown gate '{name}'", 0.0,
                           severity=gate_sev)
            )
            continue
        result = fn(cfg, artifact, data_path)
        result.severity = gate_sev
        results.append(result)
    return results


def resolve_overrides(cfg, cli_overrides=None):
    """Merge config `overrides` with CLI --override pairs (CLI wins per gate).

    Each justification must be a non-empty string for a known gate.
    Raises ValueError on anything else so a bad waiver fails closed.
    """
    merged = dict(cfg.get("overrides", {}) or {})
    for name, reason in (cli_overrides or []):
        merged[name] = reason
    for name, reason in merged.items():
        if name not in _GATES:
            raise ValueError(f"override for unknown gate '{name}'")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"override for gate '{name}' needs a non-empty justification")
        merged[name] = reason.strip()
    return merged


def exit_code(results):
    """Exit 0 unless an *error-severity* gate failed.

    A failing "warn"-severity gate is recorded (status "fail" stays in
    the report) but does not block the deployment; a "warn" status
    result was already non-blocking. Anything else (a severity that is
    not "warn") fails the run closed.
    """
    return 1 if any(r.status == FAIL and r.severity != SEVERITY_WARN
                    for r in results) else 0
