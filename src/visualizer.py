"""Matplotlib / NetworkX visualisations for the attribution engine.

All functions render headlessly (Agg backend) and return the path of the PNG
they wrote, so they work on servers and in CI without a display.
"""

from __future__ import annotations

import io
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402

from . import CURRENCY_SYMBOL  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("output")

# Palette (validated categorical slots 1 & 2 + neutral ink/surface tokens).
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"  # blue  - Last-Touch / campaigns
SERIES_2 = "#eb6834"  # orange - Markov / decay
START_COLOR = "#8a8984"
CONVERSION_COLOR = "#1baf7a"


def _style_axes(ax: plt.Axes) -> None:
    """Apply a recessive grid and quiet spines to ``ax``."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, path: Path) -> Path:
    """Save ``fig`` to ``path`` (creating parent dirs) and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    # On Windows an image viewer or IDE preview can briefly lock the old PNG,
    # so retry the write a few times before giving up.
    for attempt in range(5):
        try:
            path.write_bytes(buffer.getvalue())
            return path
        except OSError:
            if attempt == 4:
                raise
            time.sleep(0.3 * (attempt + 1))
    return path


def plot_customer_journey_graph(
    transitions: pd.DataFrame,
    top_paths: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    filename: str = "customer_journey_graph.png",
) -> Path:
    """Draw the directed journey graph of the top converting paths.

    Args:
        transitions: DataFrame with ``source``, ``target``, ``count`` columns.
        top_paths: DataFrame with ``path`` and ``conversions`` columns, listed
            beside the graph.
        output_dir: Folder to write into.
        filename: PNG file name.
    """
    graph = nx.DiGraph()
    for row in transitions.itertuples(index=False):
        graph.add_edge(row.source, row.target, weight=int(row.count))

    fig, (ax, ax_text) = plt.subplots(
        1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [2.1, 1]}, facecolor=SURFACE
    )
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    ax_text.axis("off")

    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No converting journeys found", ha="center", va="center",
                color=TEXT_SECONDARY, fontsize=12)
        return _save(fig, output_dir / filename)

    # Fixed layout: Start on the left, Conversion on the right, campaigns in
    # between, ordered left-to-right by how often they open vs. close journeys.
    campaigns = [n for n in graph.nodes if n not in ("Start", "Conversion")]
    first_hop = {n: graph["Start"][n]["weight"] if graph.has_edge("Start", n) else 0 for n in campaigns}
    last_hop = {n: graph[n]["Conversion"]["weight"] if graph.has_edge(n, "Conversion") else 0
                for n in campaigns}
    campaigns.sort(key=lambda n: (last_hop[n] - first_hop[n]))
    pos: dict[str, tuple[float, float]] = {"Start": (-2.4, 0.0), "Conversion": (2.4, 0.0)}
    n_campaigns = len(campaigns)
    for i, name in enumerate(campaigns):
        x = -1.2 + 2.4 * i / max(n_campaigns - 1, 1)
        y = 1.2 if i % 2 == 0 else -1.2
        if n_campaigns % 2 == 1 and n_campaigns >= 3 and i == n_campaigns // 2:
            y = 0.0  # odd count: centre node keeps the top/bottom rows symmetric
        pos[name] = (x, y)

    weights = [graph[u][v]["weight"] for u, v in graph.edges]
    max_w = max(weights) if weights else 1
    widths = [0.6 + 6.0 * w / max_w for w in weights]
    alphas = [0.25 + 0.65 * w / max_w for w in weights]

    loops = [(u, v) for u, v in graph.edges if u == v]
    straight = [(u, v) for u, v in graph.edges if u != v]
    for (u, v), width, alpha in zip(graph.edges, widths, alphas):
        if u == v:
            continue
        nx.draw_networkx_edges(
            graph, pos, edgelist=[(u, v)], ax=ax, width=width, alpha=alpha,
            edge_color=SERIES_1, arrows=True, arrowsize=14, arrowstyle="-|>",
            connectionstyle="arc3,rad=0.12", node_size=4200, min_target_margin=12,
        )
    for u, v in loops:
        x, y = pos[u]
        ax.annotate(f"repeat ×{graph[u][v]['weight']}", (x, y + (0.42 if y > 0 else -0.42)),
                    ha="center", va="center", fontsize=8, color=TEXT_SECONDARY)

    node_colors = [
        START_COLOR if n == "Start" else CONVERSION_COLOR if n == "Conversion" else SERIES_1
        for n in graph.nodes
    ]
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=4200,
                           edgecolors=SURFACE, linewidths=2)
    labels = {n: n.replace("_", "\n") for n in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=8,
                            font_color="white", font_weight="bold")

    edge_labels = {(u, v): graph[u][v]["weight"] for u, v in straight
                   if graph[u][v]["weight"] >= max_w * 0.15}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax, font_size=8,
                                 font_color=TEXT_PRIMARY, label_pos=0.5,
                                 connectionstyle="arc3,rad=0.12",
                                 bbox={"boxstyle": "round,pad=0.2", "fc": SURFACE, "ec": "none"})
    ax.set_title("Customer journey graph — top converting paths\n(edge width = number of journeys)",
                 color=TEXT_PRIMARY, fontsize=13, loc="left")
    ax.margins(0.08)

    lines = ["Top converting paths (conversions)", ""]
    for rank, row in enumerate(top_paths.itertuples(index=False), start=1):
        hops = row.path.replace(" > Conversion", "").split(" > ")
        lines.append(f"{rank:>2}. ({row.conversions}) " + hops[0])
        lines.extend(f"         → {hop}" for hop in hops[1:])
    ax_text.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace",
                 fontsize=8.5, color=TEXT_PRIMARY, transform=ax_text.transAxes)

    return _save(fig, output_dir / filename)


def plot_attribution_comparison(
    comparison: pd.DataFrame,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    filename: str = "attribution_comparison.png",
) -> Path:
    """Grouped bar chart of Last-Touch vs. Markov reward revenue by campaign."""
    df = comparison.sort_values("markov_revenue", ascending=False)
    campaigns = df["campaign"].tolist()
    x = range(len(campaigns))
    bar_w = 0.38

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=SURFACE)
    _style_axes(ax)
    lt = ax.bar([i - bar_w / 2 - 0.01 for i in x], df["last_touch_revenue"], bar_w,
                label="Last-Touch revenue", color=SERIES_1)
    mk = ax.bar([i + bar_w / 2 + 0.01 for i in x], df["markov_revenue"], bar_w,
                label="Markov velocity-adjusted revenue", color=SERIES_2)
    for bars in (lt, mk):
        ax.bar_label(bars, labels=[f"{CURRENCY_SYMBOL}{v:,.0f}" for v in bars.datavalues], padding=3,
                     fontsize=8, color=TEXT_SECONDARY)

    ax.set_xticks(list(x))
    ax.set_xticklabels([c.replace("_", " ") for c in campaigns], color=TEXT_PRIMARY)
    ax.set_ylabel(f"Attributed revenue ({CURRENCY_SYMBOL})", color=TEXT_SECONDARY)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter(CURRENCY_SYMBOL + "{x:,.0f}"))
    ax.set_title("Last-Touch vs. Discounted Markov attribution by campaign",
                 color=TEXT_PRIMARY, fontsize=13, loc="left")
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, loc="upper right")
    return _save(fig, output_dir / filename)


def plot_time_decay_curve(
    gamma: float = 0.85,
    base_reward: float = 100.0,
    days: int = 30,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    filename: str = "time_decay_curve.png",
) -> Path:
    """Plot ``base_reward * gamma ** t`` for ``t`` in ``[0, days]``."""
    steps = 300
    ts = [days * i / steps for i in range(steps + 1)]
    rewards = [base_reward * gamma ** t for t in ts]

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=SURFACE)
    _style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.plot(ts, rewards, color=SERIES_2, linewidth=2)
    ax.fill_between(ts, rewards, color=SERIES_2, alpha=0.08)

    for d in (1, 7, 14, 30):
        if d <= days:
            r = base_reward * gamma ** d
            ax.plot([d], [r], "o", markersize=8, color=SERIES_2, markeredgecolor=SURFACE,
                    markeredgewidth=2)
            at_edge = d >= days * 0.9
            ax.annotate(f"day {d}: {CURRENCY_SYMBOL}{r:.1f}", (d, r), textcoords="offset points",
                        xytext=(-8 if at_edge else 8, 10), ha="right" if at_edge else "left",
                        fontsize=9, color=TEXT_PRIMARY)

    if 0 < gamma < 1:
        half_life = math.log(0.5) / math.log(gamma)
        if half_life <= days:
            ax.axvline(half_life, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
            ax.annotate(f"half-life ≈ {half_life:.1f} days", (half_life, base_reward * 0.92),
                        textcoords="offset points", xytext=(6, 0), fontsize=9,
                        color=TEXT_SECONDARY)

    ax.set_xlim(0, days + 0.5)
    ax.set_ylim(0, base_reward * 1.05)
    ax.set_xlabel("Latency Δt between touch and conversion (days)", color=TEXT_SECONDARY)
    ax.set_ylabel(f"Reward per touch ({CURRENCY_SYMBOL})", color=TEXT_SECONDARY)
    ax.set_title(f"Exponential time decay: Reward = {CURRENCY_SYMBOL}{base_reward:g} × {gamma:g}^Δt",
                 color=TEXT_PRIMARY, fontsize=13, loc="left")
    return _save(fig, output_dir / filename)
