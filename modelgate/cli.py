"""Command-line interface for model-gate."""

from __future__ import annotations

import argparse
import json
import os
import sys

from modelgate import __version__
from modelgate import demo
from modelgate.baseline import BaselineError, gate_regression, load_metrics_snapshot
from modelgate.gates import (
    GATE_DESCRIPTIONS,
    GateResult,
    exit_code,
    resolve_overrides,
    run_gates,
)
from modelgate.report import render_html, render_json, render_text


def cmd_init(args):
    artifact, sample, config = demo.scaffold(args.out, broken=args.broken)
    kind = "BROKEN " if args.broken else ""
    print(f"scaffolded {kind}demo artifact:")
    print(f"  artifact  {artifact}")
    print(f"  sample    {sample}")
    print(f"  config    {config}")
    print("run: mgate check demo-model --config gates.json --data sample.csv")
    return 0


def cmd_gates(_args):
    print("available gates:")
    for name, desc in GATE_DESCRIPTIONS.items():
        print(f"  {name:<10} {desc}")
    return 0


def _parse_cli_overrides(pairs):
    """Parse --override NAME=reason pairs into (name, reason) tuples."""
    out = []
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(
                f"'{pair}' — expected --override GATE=reason (reason is required)"
            )
        name, reason = pair.split("=", 1)
        if not name.strip():
            raise ValueError(f"'{pair}' — override names a gate, e.g. --override metrics='...'")
        out.append((name.strip(), reason.strip()))
    return out


def cmd_check(args):
    if not os.path.isdir(args.artifact):
        print(f"error: artifact directory not found: {args.artifact}", file=sys.stderr)
        return 2
    if not os.path.isfile(args.config):
        print(f"error: config file not found: {args.config}", file=sys.stderr)
        return 2
    if not os.path.isfile(args.data):
        print(f"error: sample data not found: {args.data}", file=sys.stderr)
        return 2
    try:
        with open(args.config, encoding="utf-8") as f:
            cfg = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON in {args.config}: {exc}", file=sys.stderr)
        return 2

    try:
        overrides = resolve_overrides(cfg, _parse_cli_overrides(args.override))
    except ValueError as exc:
        print(f"error: invalid override: {exc}", file=sys.stderr)
        return 2

    # "regression" is not in run_gates' table (it needs a --baseline file),
    # so keep it out of the `only` filter and append the real result below.
    want_regression = args.only is not None and "regression" in args.only
    only = [g for g in args.only if g != "regression"] if args.only else args.only
    results = run_gates(cfg, args.artifact, args.data,
                        only=only, overrides=overrides)

    if args.baseline and (args.only is None or want_regression):
        results.append(
            gate_regression(cfg, args.artifact, args.data, args.baseline)
        )
    elif want_regression:
        results.append(GateResult(
            "regression", "fail",
            "regression gate needs --baseline PRIOR-REPORT.json", 0.0))

    snapshot = None
    if args.format == "json":
        try:
            snapshot = load_metrics_snapshot(args.artifact)
        except BaselineError:
            snapshot = None  # metrics gate already reports the problem

    if args.format == "json":
        out = render_json(results, os.path.basename(args.artifact.rstrip("/")),
                          metrics=snapshot)
    elif args.format == "html":
        out = render_html(results, os.path.basename(args.artifact.rstrip("/")))
    else:
        out = render_text(results, os.path.basename(args.artifact.rstrip("/")))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out + ("\n" if args.format != "html" else ""))
        print(f"wrote {args.output}")
    else:
        print(out)
    return exit_code(results)


def build_parser():
    p = argparse.ArgumentParser(
        prog="mgate",
        description="Deployment gates for model artifacts: fail the promotion, not production.",
    )
    p.add_argument("--version", action="version", version=f"mgate {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init", help="scaffold a demo model artifact")
    pi.add_argument("--out", default=".", help="directory to scaffold into (default: .)")
    pi.add_argument("--broken", action="store_true",
                    help="scaffold a demo that FAILS gates (for demos/transcripts)")
    pi.set_defaults(func=cmd_init)

    pg = sub.add_parser("gates", help="list available gate types")
    pg.set_defaults(func=cmd_gates)

    pc = sub.add_parser("check", help="run gates against a model artifact")
    pc.add_argument("artifact", help="model artifact directory")
    pc.add_argument("--config", required=True, help="gates.json config file")
    pc.add_argument("--data", required=True, help="validation sample CSV")
    pc.add_argument("--only", nargs="*", default=None,
                    help="run only these gates (default: all in config)")
    pc.add_argument("--override", action="append", default=[],
                    metavar="GATE=reason",
                    help="waive a gate for this run; a non-empty justification "
                         "is required and recorded in the report. Repeatable.")
    pc.add_argument("--baseline", default=None, metavar="PRIOR-REPORT.json",
                    help="compare metrics against a prior `check --format json` "
                         "report and fail when any metric regresses beyond its "
                         "configured tolerance (see the 'regression' config "
                         "block). Adds a 'regression' gate to the results.")
    pc.add_argument("--format", choices=["text", "json", "html"], default="text")
    pc.add_argument("-o", "--output", default=None, help="write report to file")
    pc.set_defaults(func=cmd_check)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)
