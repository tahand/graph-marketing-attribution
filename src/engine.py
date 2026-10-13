"""Attribution & analytics engine.

All path traversal and latency maths runs inside SQLite:

* A **recursive CTE** walks each converting customer's touchpoints in time
  order, one hop at a time, building the ordered path
  ``Campaign A > Campaign B > ... > Conversion``.
* Latency ``delta_t`` (days) between each touch and its conversion is computed
  with ``JULIANDAY(c.timestamp) - JULIANDAY(e.timestamp)``.
* The **Discounted Markov reward** of a touch is
  ``Base_Reward * POWER(gamma, delta_t)``.
* **Last-Touch** gives the full ``Base_Reward`` to the final touch of a path.

Both models therefore value a conversion at ``Base_Reward``; the Markov model
spreads (and time-discounts) that value over every touch in the path, so slow,
lagging journeys earn less "velocity-adjusted" revenue than fast ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .database import GraphDatabase

DEFAULT_GAMMA = 0.85
DEFAULT_BASE_REWARD = 100.0

# Shared CTE chain: conversions -> ordered touches -> recursive journey walk.
JOURNEY_CTE = """
WITH RECURSIVE
conversions AS (
    SELECT c.source_id AS customer_id,
           c.target_id AS conversion_id,
           c.timestamp
    FROM edges c
    WHERE c.relationship = 'CONVERTED_VIA'
),
ordered_touches AS (
    SELECT e.edge_id,
           e.source_id     AS customer_id,
           e.target_id     AS campaign_id,
           e.relationship,
           e.timestamp     AS touch_ts,
           c.conversion_id,
           c.timestamp     AS conversion_ts,
           JULIANDAY(c.timestamp) - JULIANDAY(e.timestamp) AS delta_days,
           ROW_NUMBER() OVER (
               PARTITION BY c.conversion_id ORDER BY e.timestamp, e.edge_id
           ) AS step,
           COUNT(*) OVER (PARTITION BY c.conversion_id) AS path_length
    FROM edges e
    JOIN conversions c ON c.customer_id = e.source_id
    WHERE e.relationship IN ('CLICKED', 'VIEWED')
      AND e.timestamp <= c.timestamp
),
journey (customer_id, conversion_id, step, path_length, edge_id, campaign_id,
         relationship, touch_ts, conversion_ts, delta_days, path) AS (
    -- Anchor: the first touch of every converting journey.
    SELECT customer_id, conversion_id, step, path_length, edge_id, campaign_id,
           relationship, touch_ts, conversion_ts, delta_days, campaign_id
    FROM ordered_touches
    WHERE step = 1
    UNION ALL
    -- Recursive hop: follow the journey to the next touch in time order.
    SELECT t.customer_id, t.conversion_id, t.step, t.path_length, t.edge_id, t.campaign_id,
           t.relationship, t.touch_ts, t.conversion_ts, t.delta_days,
           j.path || ' > ' || t.campaign_id
    FROM journey j
    JOIN ordered_touches t
      ON t.conversion_id = j.conversion_id
     AND t.step = j.step + 1
)
"""


@dataclass(frozen=True)
class AttributionConfig:
    """Parameters for the discounted reward model."""

    gamma: float = DEFAULT_GAMMA
    base_reward: float = DEFAULT_BASE_REWARD

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if self.base_reward < 0:
            raise ValueError(f"base_reward must be non-negative, got {self.base_reward}")


class AttributionEngine:
    """Runs Last-Touch and Discounted Markov attribution over the graph."""

    def __init__(self, db: GraphDatabase, config: AttributionConfig | None = None) -> None:
        self.db = db
        self.config = config or AttributionConfig()

    def _frame(self, sql: str, params: dict | tuple = ()) -> pd.DataFrame:
        """Execute ``sql`` and return the result as a DataFrame."""
        rows = self.db.query(sql, params)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])

    # ------------------------------------------------------------------ #
    # Path traversal
    # ------------------------------------------------------------------ #
    def touchpoints(self) -> pd.DataFrame:
        """Every touch on a converting path with its latency and discounted reward.

        Columns: customer_id, conversion_id, step, path_length, campaign_id,
        relationship, touch_ts, conversion_ts, delta_days, is_last_touch,
        markov_reward.
        """
        sql = JOURNEY_CTE + """
        SELECT customer_id, conversion_id, step, path_length, campaign_id, relationship,
               touch_ts, conversion_ts,
               ROUND(delta_days, 4)                               AS delta_days,
               CASE WHEN step = path_length THEN 1 ELSE 0 END     AS is_last_touch,
               :base_reward * POWER(:gamma, delta_days)           AS markov_reward
        FROM journey
        ORDER BY conversion_id, step
        """
        return self._frame(sql, {"gamma": self.config.gamma, "base_reward": self.config.base_reward})

    def customer_paths(self) -> pd.DataFrame:
        """One row per conversion with the full ordered path built by the recursive CTE.

        Columns: customer_id, conversion_id, path, path_length, journey_days.
        """
        sql = JOURNEY_CTE + """
        SELECT customer_id, conversion_id, path || ' > Conversion' AS path, path_length,
               journey_days
        FROM (
            SELECT *, ROUND(MAX(delta_days) OVER (PARTITION BY conversion_id), 3) AS journey_days
            FROM journey
        )
        WHERE step = path_length
        ORDER BY conversion_id
        """
        return self._frame(sql)

    def top_paths(self, n: int = 10) -> pd.DataFrame:
        """Most frequent converting paths with count and average journey length."""
        paths = self.customer_paths()
        if paths.empty:
            return pd.DataFrame(columns=["path", "conversions", "avg_journey_days"])
        return (
            paths.groupby("path")
            .agg(conversions=("conversion_id", "count"), avg_journey_days=("journey_days", "mean"))
            .sort_values(["conversions", "avg_journey_days"], ascending=[False, True])
            .head(n)
            .reset_index()
        )

    # ------------------------------------------------------------------ #
    # Attribution models
    # ------------------------------------------------------------------ #
    def last_touch_attribution(self) -> pd.DataFrame:
        """Credit the full Base_Reward of each conversion to its final touch.

        Columns: campaign, last_touch_conversions, last_touch_revenue.
        """
        sql = JOURNEY_CTE + """
        SELECT campaign_id                 AS campaign,
               COUNT(*)                    AS last_touch_conversions,
               COUNT(*) * :base_reward     AS last_touch_revenue
        FROM journey
        WHERE step = path_length
        GROUP BY campaign_id
        """
        return self._frame(sql, {"base_reward": self.config.base_reward})

    def markov_attribution(self) -> pd.DataFrame:
        """Discounted Markov reward: each touch earns Base_Reward * gamma ^ delta_t.

        Columns: campaign, total_conversions (conversions whose path includes the
        campaign), touches, avg_latency_days, markov_revenue.
        """
        sql = JOURNEY_CTE + """
        SELECT campaign_id                                     AS campaign,
               COUNT(DISTINCT conversion_id)                   AS total_conversions,
               COUNT(*)                                        AS touches,
               AVG(delta_days)                                 AS avg_latency_days,
               SUM(:base_reward * POWER(:gamma, delta_days))   AS markov_revenue
        FROM journey
        GROUP BY campaign_id
        """
        return self._frame(sql, {"gamma": self.config.gamma, "base_reward": self.config.base_reward})

    def compare_models(self) -> pd.DataFrame:
        """Side-by-side Last-Touch vs. Discounted Markov results per campaign.

        Includes every Campaign node, so campaigns that never closed a
        conversion still appear with zero last-touch revenue.
        """
        campaigns = self._frame("SELECT node_id AS campaign FROM nodes WHERE label = 'Campaign'")
        if campaigns.empty:
            return pd.DataFrame()
        markov = self.markov_attribution()
        last = self.last_touch_attribution()

        df = campaigns
        for part in (markov, last):
            if not part.empty:
                df = df.merge(part, on="campaign", how="left")
        for col in ("total_conversions", "touches", "avg_latency_days", "markov_revenue",
                    "last_touch_conversions", "last_touch_revenue"):
            if col not in df:
                df[col] = 0.0
        df = df.fillna(0)
        df[["total_conversions", "touches", "last_touch_conversions"]] = df[
            ["total_conversions", "touches", "last_touch_conversions"]
        ].astype(int)

        lt_total = df["last_touch_revenue"].sum()
        mk_total = df["markov_revenue"].sum()
        df["last_touch_share"] = df["last_touch_revenue"] / lt_total if lt_total else 0.0
        df["markov_share"] = df["markov_revenue"] / mk_total if mk_total else 0.0
        return df.sort_values("markov_revenue", ascending=False).reset_index(drop=True)

    def transition_counts(self, max_paths: int | None = None) -> pd.DataFrame:
        """Aggregate step-to-step transitions (incl. Start / Conversion) over converting paths.

        Args:
            max_paths: if given, only use the ``max_paths`` most frequent paths.

        Columns: source, target, count.
        """
        paths = self.top_paths(max_paths) if max_paths else (
            self.customer_paths().groupby("path").size().rename("conversions").reset_index()
        )
        counts: dict[tuple[str, str], int] = {}
        for path, n in zip(paths["path"], paths["conversions"]):
            hops = ["Start"] + path.split(" > ")
            for a, b in zip(hops, hops[1:]):
                counts[(a, b)] = counts.get((a, b), 0) + int(n)
        return pd.DataFrame(
            [(a, b, n) for (a, b), n in counts.items()], columns=["source", "target", "count"]
        )
