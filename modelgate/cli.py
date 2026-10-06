"""Command-line interface for model-gate."""

from __future__ import annotations

import argparse
import json
import os
import sys

from modelgate import __version__
from modelgate import demo
from modelgate.gates import (
    GATE_DESCRIPTIONS,
    exit_code,
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

    results = run_gates(cfg, args.artifact, args.data, only=args.only)

    if args.format == "json":
        out = render_json(results, os.path.basename(args.artifact.rstrip("/")))
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
    pc.add_argument("--format", choices=["text", "json", "html"], default="text")
    pc.add_argument("-o", "--output", default=None, help="write report to file")
    pc.set_defaults(func=cmd_check)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)
