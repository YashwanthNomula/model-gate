"""model-gate: CI deployment gates for model artifacts.

Validate a model artifact directory before promotion to production:
required files, schema conformance, metric thresholds, latency budget,
artifact size, and prediction contract. Zero dependencies.
"""

__version__ = "1.0.0"
