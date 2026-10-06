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
| `metrics` | Each metric in `metrics.json` ≥ its configured threshold | A configured metric is missing or below threshold |
| `latency` | Imports the entrypoint via `importlib`, calls `predict` on batches `n_calls` times; reports p50/p95/p99 | p95 > `p95_ms` (warns within 20% of budget) |
| `size` | Artifact file (`size.file`) ≤ `size.max_bytes` | File missing or over the limit |
| `contract` | `predict` on the first `n_rows` sample rows: output count matches input, every label ∈ `manifest.json` labels | Length mismatch or an unknown label |

Exit code: **0** if no gate fails (warnings allowed), **1** if any gate fails —
so `mgate check ... || exit 1` gates the promotion step.

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

## Config reference (`gates.json`)

```json
{
  "gates": ["files", "schema", "metrics", "latency", "size", "contract"],
  "files":   {"required": ["manifest.json", "model.py", "schema.json", "metrics.json", "model.bin"]},
  "schema":  {"max_missing_rate": 0.01},
  "metrics": {"accuracy": 0.90, "f1": 0.80, "auc": 0.90},
  "latency": {"n_calls": 10, "batch": 100, "p95_ms": 50.0},
  "size":    {"file": "model.bin", "max_bytes": 65536},
  "contract":{"n_rows": 50}
}
```

See [`examples/`](examples/) for a ready-made `gates.json` and `sample.csv`.

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
mgate check <artifact> --config gates.json --data sample.csv [--format text|json|html] [-o file] [--only GATES...]
mgate gates                                       list available gate types
```

## Tests

```bash
python3 -m pytest tests/ -q    # 31 tests
```

Pure Python, zero dependencies.
