#!/usr/bin/env python
"""End-to-end demo of the Multi-Touch Marketing Attribution Engine.

Steps:
    a) Reset / initialise ``marketing_graph.db``.
    b) Generate synthetic customer-journey graphs.
    c) Run Last-Touch and Discounted Markov attribution.
    d) Print a campaign summary table.
    e) Render three PNG visualisations into ``output/``.

Usage:
    python run_demo.py [--customers 500] [--gamma 0.85] [--base-reward 100] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _check_dependencies() -> None:
    """Exit with a helpful message if a required package is missing."""
    missing = []
    for module in ("pandas", "networkx", "matplotlib", "tabulate"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        sys.exit(
            f"Missing required package(s): {', '.join(missing)}\n"
            f"Install them with:  pip install -r requirements.txt"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--customers", type=int, default=500, help="customers to simulate (default 500)")
    parser.add_argument("--gamma", type=float, default=0.85, help="daily discount factor (default 0.85)")
    parser.add_argument("--base-reward", type=float, default=100.0, help="reward per conversion in GBP (default 100)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed; use -1 for a random run")
    parser.add_argument("--db", type=Path, default=ROOT / "marketing_graph.db", help="SQLite file path")
    parser.add_argument("--output", type=Path, default=ROOT / "output", help="folder for PNG charts")
    return parser.parse_args(argv)


def print_summary(comparison, top_paths, gamma: float) -> None:
    """Print the campaign comparison table and the most common paths."""
    from tabulate import tabulate

    from src import CURRENCY_SYMBOL

    rows = [
        [
            r.campaign,
            r.total_conversions,
            f"{r.avg_latency_days:.2f}",
            f"{CURRENCY_SYMBOL}{r.last_touch_revenue:,.2f}",
            f"{CURRENCY_SYMBOL}{r.markov_revenue:,.2f}",
            f"{(r.markov_share - r.last_touch_share) * 100:+.1f} pp",
        ]
        for r in comparison.itertuples(index=False)
    ]
    rows.append([
        "TOTAL",
        "",
        "",
        f"{CURRENCY_SYMBOL}{comparison['last_touch_revenue'].sum():,.2f}",
        f"{CURRENCY_SYMBOL}{comparison['markov_revenue'].sum():,.2f}",
        "",
    ])
    headers = [
        "Campaign Name",
        "Total Conversions",
        "Avg Latency (Days)",
        "Last-Touch Revenue (GBP)",
        f"Markov Velocity-Adj. Revenue (GBP, γ={gamma:g})",
        "Share Shift",
    ]
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline",
                   colalign=("left", "right", "right", "right", "right", "right")))
    print("  Total Conversions = conversions whose path touched the campaign (multi-touch).")
    print("  Share Shift = Markov share of credit minus Last-Touch share of credit.\n")

    print("Top converting journey paths:")
    print(tabulate(
        [[r.path, r.conversions, f"{r.avg_journey_days:.1f}"] for r in top_paths.itertuples(index=False)],
        headers=["Path", "Conversions", "Avg Journey (Days)"],
        tablefmt="simple",
    ))


def main(argv: list[str] | None = None) -> int:
    """Run the full demo pipeline. Returns a process exit code."""
    # Windows consoles often default to a legacy code page that cannot print
    # the table's box-drawing characters or the γ symbol.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    _check_dependencies()
    args = parse_args(argv)

    from src.database import DatabaseError, GraphDatabase
    from src.engine import AttributionConfig, AttributionEngine
    from src.generator import generate_dataset
    from src import visualizer

    started = time.perf_counter()
    try:
        config = AttributionConfig(gamma=args.gamma, base_reward=args.base_reward)
        seed = None if args.seed == -1 else args.seed

        with GraphDatabase(args.db) as db:
            print(f"[1/5] Resetting knowledge graph at {args.db} ...")
            db.reset()

            print(f"[2/5] Generating {args.customers} synthetic customer journeys ...")
            summary = generate_dataset(db, n_customers=args.customers, seed=seed)
            counts = db.counts()
            print(
                f"      {summary.customers} customers, {summary.touchpoints} touchpoints, "
                f"{summary.conversions} conversions ({summary.conversion_rate:.0%})"
            )
            print(f"      latency mix: {summary.latency_profiles}")
            print(f"      graph: {counts}")

            print("[3/5] Running Last-Touch and Discounted Markov attribution ...")
            engine = AttributionEngine(db, config)
            comparison = engine.compare_models()
            top_paths = engine.top_paths(10)
            transitions = engine.transition_counts(max_paths=15)

        if comparison.empty:
            print("No conversions were generated - nothing to attribute.")
            return 1

        print("[4/5] Attribution summary\n")
        print_summary(comparison, top_paths, config.gamma)

        print("\n[5/5] Rendering visualisations ...")
        outputs = [
            visualizer.plot_customer_journey_graph(transitions, top_paths, args.output),
            visualizer.plot_attribution_comparison(comparison, args.output),
            visualizer.plot_time_decay_curve(config.gamma, config.base_reward, 30, args.output),
        ]
        for path in outputs:
            print(f"      wrote {path}")
    except (DatabaseError, ValueError, OSError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    print(f"\nDone in {time.perf_counter() - started:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
