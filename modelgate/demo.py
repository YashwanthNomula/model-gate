"""Scaffold a demo model artifact + validation sample.

`mgate init` writes a *passing* demo; `mgate init --broken` writes one that
fails two gates (f1 under threshold, schema column missing from the sample),
so the README can show both sides of a deployment gate.
"""

from __future__ import annotations

import json
import os
import random

MODEL_TEMPLATE = '''"""Demo fraud model: a hand-written linear scorer, zero dependencies.

Exposes predict(records: list[dict]) -> list[str] exactly as a real serving
entrypoint would.
"""

BIAS = -2.2
W_AMOUNT = 0.004
W_NIGHT = 0.9
W_MERCHANT = {
    "grocery": -0.3,
    "gas": -0.2,
    "travel": 0.4,
    "electronics": 0.5,
    "crypto": 1.2,
    "other": 0.0,
}
W_DEVICE = {"mobile": 0.1, "desktop": -0.1, "atm": 0.3}


def _sigmoid(x):
    return 1.0 / (1.0 + 2.718281828459045 ** (-x))


def score(record):
    z = BIAS + W_AMOUNT * float(record["amount"])
    hour = int(record["hour"])
    if hour < 6 or hour >= 23:
        z += W_NIGHT
    z += W_MERCHANT.get(record["merchant"], W_MERCHANT["other"])
    z += W_DEVICE.get(record["device"], 0.0)
    return _sigmoid(z)


def predict(records):
    """records: list of dicts with amount/merchant/hour/device -> list of labels."""
    return ["fraud" if score(r) > 0.5 else "legit" for r in records]
'''

SCHEMA = {"amount": "float", "merchant": "string", "hour": "int", "device": "string"}
LABELS = ["legit", "fraud"]

GATES_CONFIG = {
    "gates": ["files", "schema", "metrics", "latency", "size", "contract"],
    "files": {
        "required": ["manifest.json", "model.py", "schema.json", "metrics.json", "model.bin"]
    },
    "schema": {"max_missing_rate": 0.01},
    "metrics": {"accuracy": 0.90, "f1": 0.80, "auc": 0.90},
    "latency": {"n_calls": 10, "batch": 100, "p95_ms": 50.0},
    "size": {"file": "model.bin", "max_bytes": 65536},
    "contract": {"n_rows": 50},
}

MERCHANTS = ["grocery", "gas", "travel", "electronics", "crypto", "other"]
DEVICES = ["mobile", "desktop", "atm"]
COLUMNS = ["amount", "merchant", "hour", "device"]


def _sample_rows(seed=7, n=200, columns=COLUMNS):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        row = {
            "amount": f"{rng.uniform(1, 5000):.2f}",
            "merchant": rng.choice(MERCHANTS),
            "hour": str(rng.randint(0, 23)),
            "device": rng.choice(DEVICES),
        }
        rows.append([row[c] for c in columns])
    return rows


def build_artifact(path, broken=False):
    """Write the demo model artifact directory."""
    os.makedirs(path, exist_ok=True)
    schema = dict(SCHEMA)
    metrics = {"accuracy": 0.93, "f1": 0.87, "auc": 0.95}
    if broken:
        # schema declares a field the sample data does not have...
        schema["country"] = "string"
        # ...and f1 misses the configured threshold.
        metrics = {"accuracy": 0.91, "f1": 0.72, "auc": 0.90}
    manifest = {
        "name": "demo-fraud-model",
        "version": "0.1.0",
        "entrypoint": "model.py",
        "schema": "schema.json",
        "metrics": "metrics.json",
        "labels": LABELS,
    }
    with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(path, "model.py"), "w", encoding="utf-8") as f:
        f.write(MODEL_TEMPLATE)
    with open(os.path.join(path, "schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    with open(os.path.join(path, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(path, "model.bin"), "wb") as f:
        f.write(os.urandom(4096))  # dummy weights blob
    return path


def build_sample(path, broken=False):
    """Write the validation sample CSV."""
    rows = _sample_rows(columns=COLUMNS)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(COLUMNS) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")
    return path


def build_config(path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(GATES_CONFIG, f, indent=2)
    return path


def scaffold(out_dir=".", broken=False):
    """Create demo-model/, sample.csv and gates.json under out_dir."""
    artifact = build_artifact(os.path.join(out_dir, "demo-model"), broken=broken)
    sample = build_sample(os.path.join(out_dir, "sample.csv"))
    config = build_config(os.path.join(out_dir, "gates.json"))
    return artifact, sample, config
