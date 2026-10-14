"""Synthetic multi-touch customer-journey generator.

Each simulated customer gets a sequence of touchpoints (VIEWED / CLICKED edges
from the Customer node to a Campaign node). Converting customers additionally
get a CONVERTED_VIA edge to their own Conversion node.

Journeys follow a simple funnel model: awareness campaigns (Organic_Blog,
LinkedIn_Video_Ad) tend to appear early in a path, while closing campaigns
(Retargeting_Banner, Google_Search_Branded) tend to appear late. Conversion
latencies are drawn from a mixture of fast (1-2 day), medium (3-14 day) and
slow lagging (30-60 day) profiles.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from . import CURRENCY_CODE
from .database import GraphDatabase

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"
SIMULATION_START = datetime(2026, 1, 1, 8, 0, 0)
SIMULATION_WINDOW_DAYS = 90

CHANNELS: dict[str, dict[str, str]] = {
    "channel:social": {"name": "Paid Social"},
    "channel:paid_search": {"name": "Paid Search"},
    "channel:display": {"name": "Display"},
    "channel:email": {"name": "Email"},
    "channel:organic": {"name": "Organic Content"},
}

# stage: 0.0 = top-of-funnel awareness, 1.0 = bottom-of-funnel closer.
# view_prob: probability a touch is a passive VIEW rather than a CLICK.
# popularity: relative reach of the campaign.
# cost_per_touch: media cost per impression/click in GBP.
CAMPAIGNS: dict[str, dict] = {
    "LinkedIn_Video_Ad": {
        "channel_id": "channel:social", "stage": 0.10, "view_prob": 0.70,
        "popularity": 1.0, "cost_per_touch": 4.20,
    },
    "Organic_Blog": {
        "channel_id": "channel:organic", "stage": 0.20, "view_prob": 0.45,
        "popularity": 1.1, "cost_per_touch": 0.00,
    },
    "Email_Newsletter": {
        "channel_id": "channel:email", "stage": 0.55, "view_prob": 0.35,
        "popularity": 0.9, "cost_per_touch": 0.15,
    },
    "Retargeting_Banner": {
        "channel_id": "channel:display", "stage": 0.80, "view_prob": 0.80,
        "popularity": 0.9, "cost_per_touch": 1.10,
    },
    "Google_Search_Branded": {
        "channel_id": "channel:paid_search", "stage": 0.95, "view_prob": 0.10,
        "popularity": 1.0, "cost_per_touch": 2.60,
    },
}

# (profile name, probability, min latency days, max latency days)
LATENCY_PROFILES: tuple[tuple[str, float, float, float], ...] = (
    ("fast", 0.40, 1.0, 2.0),
    ("medium", 0.30, 3.0, 14.0),
    ("slow", 0.30, 30.0, 60.0),
)

TOUCH_COUNT_WEIGHTS = {1: 0.12, 2: 0.22, 3: 0.26, 4: 0.20, 5: 0.12, 6: 0.08}
SEGMENTS = ("SMB", "Mid-Market", "Enterprise")
REGIONS = ("NA", "EMEA", "APAC", "LATAM")


@dataclass
class GenerationSummary:
    """Counts describing a generated dataset."""

    customers: int = 0
    conversions: int = 0
    touchpoints: int = 0
    latency_profiles: dict[str, int] = field(default_factory=dict)

    @property
    def conversion_rate(self) -> float:
        """Share of customers who converted."""
        return self.conversions / self.customers if self.customers else 0.0


def _fmt(ts: datetime) -> str:
    """Format a datetime as a SQLite-friendly ISO-8601 string."""
    return ts.strftime(TIMESTAMP_FORMAT)


def _pick_campaign(rng: random.Random, position: float) -> str:
    """Pick a campaign whose funnel stage best matches ``position`` (0..1)."""
    names = list(CAMPAIGNS)
    weights = [
        CAMPAIGNS[n]["popularity"] * math.exp(-((CAMPAIGNS[n]["stage"] - position) ** 2) / 0.12)
        for n in names
    ]
    return rng.choices(names, weights=weights, k=1)[0]


def _pick_latency(rng: random.Random) -> tuple[str, float]:
    """Draw a (profile, journey length in days) pair from the latency mixture."""
    names = [p[0] for p in LATENCY_PROFILES]
    probs = [p[1] for p in LATENCY_PROFILES]
    name = rng.choices(names, weights=probs, k=1)[0]
    _, _, lo, hi = next(p for p in LATENCY_PROFILES if p[0] == name)
    return name, rng.uniform(lo, hi)


def _touch_times(rng: random.Random, start: datetime, end: datetime, n: int) -> list[datetime]:
    """Return ``n`` sorted touch times: first at ``start``, last at ``end``."""
    if n == 1:
        return [end]
    span = (end - start).total_seconds()
    middle = sorted(rng.uniform(0, span) for _ in range(n - 2))
    return [start] + [start + timedelta(seconds=s) for s in middle] + [end]


def generate_dataset(
    db: GraphDatabase,
    n_customers: int = 500,
    conversion_rate: float = 0.6,
    seed: int | None = 42,
) -> GenerationSummary:
    """Generate synthetic customer journeys and write them into ``db``.

    Args:
        db: An initialised :class:`GraphDatabase`.
        n_customers: Number of Customer nodes to simulate.
        conversion_rate: Probability that a customer converts.
        seed: RNG seed for reproducible output (``None`` for random).

    Returns:
        A :class:`GenerationSummary` with counts of what was written.
    """
    if n_customers <= 0:
        raise ValueError("n_customers must be positive")
    if not 0.0 <= conversion_rate <= 1.0:
        raise ValueError("conversion_rate must be between 0 and 1")

    rng = random.Random(seed)
    summary = GenerationSummary(latency_profiles={p[0]: 0 for p in LATENCY_PROFILES})

    nodes: list[tuple[str, str, dict]] = []
    edges: list[tuple[str, str, str, str, dict]] = []

    for channel_id, props in CHANNELS.items():
        nodes.append((channel_id, "Channel", props))
    for name, cfg in CAMPAIGNS.items():
        nodes.append((name, "Campaign", {
            "name": name,
            "channel_id": cfg["channel_id"],
            "funnel_stage": cfg["stage"],
            "cost_per_touch": cfg["cost_per_touch"],
            "currency": CURRENCY_CODE,
        }))

    touch_counts = list(TOUCH_COUNT_WEIGHTS)
    touch_weights = list(TOUCH_COUNT_WEIGHTS.values())

    for i in range(1, n_customers + 1):
        customer_id = f"customer:{i:04d}"
        nodes.append((customer_id, "Customer", {
            "segment": rng.choice(SEGMENTS),
            "region": rng.choice(REGIONS),
        }))
        summary.customers += 1

        n_touches = rng.choices(touch_counts, weights=touch_weights, k=1)[0]
        first_touch = SIMULATION_START + timedelta(days=rng.uniform(0, SIMULATION_WINDOW_DAYS))
        converts = rng.random() < conversion_rate

        if converts:
            profile, latency_days = _pick_latency(rng)
            summary.latency_profiles[profile] += 1
            conversion_ts = first_touch + timedelta(days=latency_days)
            # The final touch usually lands shortly before conversion, but can lag.
            last_gap = min(latency_days * 0.5, rng.expovariate(1.0 / 0.75))
            last_touch = conversion_ts - timedelta(days=max(last_gap, 0.01))
            if n_touches == 1:
                # Single-touch journeys: the one touch defines the latency.
                last_touch = first_touch
        else:
            profile, conversion_ts = None, None
            last_touch = first_touch + timedelta(days=rng.uniform(1, 45))

        times = _touch_times(rng, first_touch, last_touch, n_touches)
        for idx, ts in enumerate(times):
            position = 1.0 if n_touches == 1 and converts else idx / max(n_touches - 1, 1)
            campaign = _pick_campaign(rng, position)
            cfg = CAMPAIGNS[campaign]
            relationship = "VIEWED" if rng.random() < cfg["view_prob"] else "CLICKED"
            edges.append((customer_id, campaign, relationship, _fmt(ts), {
                "channel_id": cfg["channel_id"],
                "touch_index": idx + 1,
                "cost": cfg["cost_per_touch"],
                "currency": CURRENCY_CODE,
            }))
            summary.touchpoints += 1

        if converts:
            conversion_id = f"conversion:{i:04d}"
            nodes.append((conversion_id, "Conversion", {
                "type": "purchase",
                "latency_profile": profile,
            }))
            edges.append((customer_id, conversion_id, "CONVERTED_VIA", _fmt(conversion_ts), {
                "touches_before_conversion": n_touches,
            }))
            summary.conversions += 1

    db.add_nodes(nodes)
    db.add_edges(edges)
    return summary
