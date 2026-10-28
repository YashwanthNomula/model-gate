# model-gate

**Fail the promotion, not production.** A CI deployment gate that validates a model artifact
before it reaches production: required files, schema conformance, metric thresholds, latency
budget, artifact size, and prediction contract. Zero dependencies, runs in seconds, exits
non-zero on failure so it drops straight into any CI/CD pipeline.

```bash
pip install -e .
mgate init                                        # scaffold a demo artifact
mgate check demo-model --config gates.json --data sample.csv || exit 1
```

## Demo

Real output, captured from an actual run:

```
$ mgate init
scaffolded demo artifact:
  artifact  ./demo-model
  sample    ./sample.csv
  config    ./gates.json
run: mgate check demo-model --config gates.json --data sample.csv

$ mgate check demo-model --config gates.json --data sample.csv
model-gate  artifact: demo-model

  ✓ files      pass  5 required files present
  ✓ schema     pass  200 rows x 4 fields ok, missing rate 0.000%
  ✓ metrics    pass  accuracy 0.93 ≥ 0.9, f1 0.87 ≥ 0.8, auc 0.95 ≥ 0.9
  ✓ latency    pass  p50 0.08ms, p95 0.11ms, p99 0.11ms over 10x100 predict calls
  ✓ size       pass  model.bin 4.0KB (limit 64.0KB)
  ✓ contract   pass  50 rows in, 50 labels out, all ∈ ['legit', 'fraud']

6/6 passed
$ echo $?
0
```

And the failing side — `mgate init --broken` scaffolds an artifact with an f1 under
threshold and a schema column missing from the sample data:

```
$ mgate init --broken --out broken
scaffolded BROKEN demo artifact:
  artifact  broken/demo-model
  sample    broken/sample.csv
  config    broken/gates.json

$ mgate check broken/demo-model --config broken/gates.json --data broken/sample.csv
model-gate  artifact: demo-model

  ✓ files      pass  5 required files present
  ✗ schema     fail  missing columns: country
  ✗ metrics    fail  below threshold: f1 0.72 < 0.8
  ✓ latency    pass  p50 0.09ms, p95 0.11ms, p99 0.11ms over 10x100 predict calls
  ✓ size       pass  model.bin 4.0KB (limit 64.0KB)
  ✓ contract   pass  50 rows in, 50 labels out, all ∈ ['legit', 'fraud']

4/6 passed (2 failed)
$ echo $?
1
```

Machine-readable reports for pipeline consumers:

```bash
mgate check demo-model --config gates.json --data sample.csv --format json
mgate check demo-model --config gates.json --data sample.csv --format html -o report.html
```

## The gates

| Gate | What it checks | Fails when |
|---|---|---|
| `files` | Every file in `files.required` exists in the artifact dir | Any required file is missing |
| `schema` | Sample CSV has every `schema.json` field, values coerce to declared types (`int`/`float`/`string`/`bool`), missing rate ≤ `max_missing_rate` (default 1%) | Missing column, type errors, or missing rate over the limit. Extra columns are fine (noted) |
| `metrics` | Each metric in `metrics.json` meets its configured threshold — `≥` by default, `≤` when `direction` is `lower_is_better` (see Metric direction) | A configured metric is missing, breaches its threshold, or has an invalid `direction` |
| `latency` | Imports the entrypoint via `importlib`, calls `predict` on batches `n_calls` times; reports p50/p95/p99 | p95 > `p95_ms` (warns within 20% of budget) |
| `size` | Artifact file (`size.file`) ≤ `size.max_bytes` | File missing or over the limit |
| `contract` | `predict` on the first `n_rows` sample rows: output count matches input, every label ∈ `manifest.json` labels | Length mismatch or an unknown label |
| `regression` | Current `metrics.json` vs a prior JSON report (`--baseline`): no metric regresses beyond its tolerance (direction-aware; see Baseline regression comparison) | Any metric regresses beyond tolerance, a baseline metric goes missing, or the baseline/config is malformed |

Exit code: **0** if no *error-severity* gate fails (warnings allowed), **1** if any
error-severity gate fails — so `mgate check ... || exit 1` gates the promotion step.
Every gate defaults to severity `"error"`; declare `"warn"` per gate to record a
failure without blocking (see Warning-severity gates).

## Gate overrides (exception workflow)

Production gates need a break-glass path. Any gate can be waived — either in
`gates.json` or per-run on the command line — but every waiver requires a
non-empty justification, and that justification is recorded in the report:

```json
{
  "overrides": {
    "metrics": "f1 dip approved by on-call during incident #4821; retrain scheduled"
  }
}
```

```bash
mgate check demo-model --config gates.json --data sample.csv \
  --override "metrics=f1 dip approved by on-call during incident #4821"
```

A waived gate is **not executed** — it reports `skip` with the reason in the
detail, visible in the terminal, JSON, and HTML reports. Skips don't fail the
run, so `mgate check ... || exit 1` still passes with a recorded waiver.
Overrides for unknown gates or with an empty justification are rejected
(exit code 2): a waiver that names the wrong gate is a config bug, not a pass.

```
  – metrics    skip  waived by override: f1 dip approved by on-call during incident #4821
```

## CI usage

```yaml
# .github/workflows/promote.yml
- name: Gate model promotion
  run: |
    pip install ./model-gate
    mgate check artifacts/candidate \
      --config ci/gates.json \
      --data ci/validation-sample.csv \
      --format json -o gate-report.json || exit 1
```

Gate a single gate subset in dev: `mgate check demo-model --config gates.json --data sample.csv --only schema metrics`.

Gate selection fails closed: an unknown name in `--only` (or in the config
`gates` list), and an empty selection, are rejected with exit code 2 — a
typo like `--only schem` must never report a pass for gates that never ran.

## Config reference (`gates.json`)

```json
{
  "gates": ["files", "schema", "metrics", "latency", "size", "contract"],
  "files":   {"required": ["manifest.json", "model.py", "schema.json", "metrics.json", "model.bin"]},
  "schema":  {"max_missing_rate": 0.01},
  "metrics": {"accuracy": 0.90, "f1": 0.80, "auc": 0.90},
  "latency": {"n_calls": 10, "batch": 100, "p95_ms": 50.0},
  "size":    {"file": "model.bin", "max_bytes": 65536},
  "contract":{"n_rows": 50},
  "overrides": {"metrics": "required justification — waives the gate, recorded as skip"},
  "severity": {"metrics": "warn", "latency": "warn"}
}
```

See [`examples/`](examples/) for a ready-made `gates.json` and `sample.csv`.

### Metric direction (`higher_is_better` | `lower_is_better`)

Not every metric improves upward. Loss, p95 latency, and error rates should go
**down**, so the `metrics` gate accepts a per-metric `direction`:

```json
"metrics": {
  "accuracy": 0.90,
  "loss": {"threshold": 0.50, "direction": "lower_is_better"},
  "p95_latency_ms": {"threshold": 100.0, "direction": "lower_is_better"}
}
```

A bare number keeps the default `higher_is_better` (value `≥` threshold) and is
fully backwards compatible. With `lower_is_better` the gate fails when the
value `>` the threshold. Omitting `direction` in the object form also defaults
to `higher_is_better`; an unknown direction fails the gate closed rather than
guessing. The direction appears in every report (terminal, JSON, HTML), e.g.
`✓ metrics pass loss 0.32 ≤ 0.5 (lower is better)`, so an audit shows which
comparison each threshold used.

### Warning-severity gates

Some gates are worth tracking without blocking the promotion. Each gate can be
declared `severity: "warn"` in the config (default `"error"`, so existing
configs behave exactly as before):

```json
"severity": {"metrics": "warn", "latency": "warn"}
```

A failing `"warn"` gate still records status `fail` — the terminal row, JSON
gate entry (with its `severity` field), and HTML row all show the failure
honestly — but it does **not** fail the deployment: the overall verdict stays
PASS and the exit code stays 0. The reports call the warnings out explicitly:

- Terminal: a `WARNINGS — deployment proceeds, but these gates need attention:`
  section after the summary (summary counts them separately, e.g.
  `4/6 passed (1 failed, 1 warning)`).
- JSON: a top-level `"warnings"` list of the warning gate entries.
- HTML: a yellow `PASS (with warnings)` banner instead of the green one.

`"error"` gates keep the current semantics (exit 1). For a run where you want
everything enforced — e.g. the final prod promotion — `mgate check --strict`
promotes all warn-severity gates to error for that run only. Unknown severity
strings, unknown gate names, and non-object `severity` blocks are rejected
(exit code 2): a severity that names the wrong gate is a config bug, not a pass.

With `"severity": {"metrics": "warn"}` on the broken demo (where `schema` stays
`"error"`), the warn failure is recorded but only the error failure blocks:

```
$ mgate check broken/demo-model --config broken/gates.json --data broken/sample.csv
model-gate  artifact: demo-model

  ✓ files      pass  5 required files present
  ✗ schema     fail  missing columns: country
  ✗ metrics    fail  below threshold: f1 0.72 < 0.8 [severity: warn]
  ✓ latency    pass  p50 0.08ms, p95 0.14ms, p99 0.15ms over 10x100 predict calls
  ✓ size       pass  model.bin 4.0KB (limit 64.0KB)
  ✓ contract   pass  50 rows in, 50 labels out, all ∈ ['legit', 'fraud']

4/6 passed (1 failed, 1 warning)

WARNINGS — deployment proceeds, but these gates need attention:
  ! metrics    fail  below threshold: f1 0.72 < 0.8 [severity: warn]
$ echo $?
1
```

Had `schema` been declared `"warn"` too, the same run would exit 0 with
`4/6 passed (2 warning)`.

### Baseline regression comparison

Absolute thresholds catch a bad model; they miss a good model quietly getting
worse. `mgate check --baseline PRIOR-REPORT.json` adds a `regression` gate
that compares the current `metrics.json` against the metrics snapshot stored
in a previous JSON report and fails when any metric regresses beyond its
tolerance:

```bash
# After promoting v1: keep its JSON report as the baseline.
mgate check demo-model --config gates.json --data sample.csv --format json -o v1-report.json
# Later, when v2 is a candidate:
mgate check demo-model-v2 --config gates.json --data sample.csv --baseline v1-report.json
```

Tolerances live in the config, alongside the other gate settings:

```json
"regression": {
  "default_tolerance": 0.02,
  "tolerances": {"accuracy": 0.01, "loss": 0.005}
}
```

Direction is inherited from the `metrics` gate specs, so a `lower_is_better`
metric regresses when it *rises*. A metric with no per-metric tolerance and
no default is skipped, never guessed at, and a baseline metric missing from
the current run (or a non-numeric value, or an unreadable baseline file)
fails the gate closed. A run where every comparison is skipped reports `warn`
— visible, but exit 0. The regression gate also honours `--only regression`.

## Model artifact layout

```
demo-model/
  manifest.json   {"name","version","entrypoint":"model.py","schema":"schema.json",
                   "metrics":"metrics.json","labels":["legit","fraud"]}
  model.py        exposes predict(records: list[dict]) -> list[str]
  schema.json     {"amount":"float","merchant":"string","hour":"int","device":"string"}
  metrics.json    {"accuracy":0.93,"f1":0.87,"auc":0.95}
  model.bin       artifact bytes (e.g. weights)
```

## Commands

```
mgate init [--out DIR] [--broken]                 scaffold a demo (passing, or failing with --broken)
mgate check <artifact> --config gates.json --data sample.csv [--format text|json|html] [-o file] [--only GATES...] [--override GATE=reason ...] [--baseline PRIOR-REPORT.json] [--strict]
mgate gates                                       list available gate types
```

## Tests

```bash
python3 -m pytest tests/ -q    # 92 tests
```

Pure Python, zero dependencies.
