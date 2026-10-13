# graph-marketing-attribution

Zero-cost Python & SQLite data pipeline converting multi-touch marketing journeys into an edge Knowledge Graph for Discounted Markov Attribution.

It runs locally with no servers or paid services. Customer journeys are stored as a property graph in a single SQLite file. The paths are walked with recursive CTEs, and each campaign gets credit from two models:

| Model | Credit rule |
|---|---|
| **Last-Touch** | The final touch before a conversion gets the full `Base_Reward`. |
| **Discounted Markov (velocity-adjusted)** | Every touch on the path earns `Base_Reward × γ^Δt`, where `Δt` is the number of days between that touch and the conversion. |

The defaults are `γ = 0.85` and `Base_Reward = 100.0`. That gives a half-life of about 4.3 days: a touch one day before a conversion earns 85, a touch one week before earns 32, and a touch a month before earns less than 1.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_demo.py
```

The demo:

1. Resets `marketing_graph.db`.
2. Generates 500 synthetic customers across 5 campaigns.
3. Runs both attribution models.
4. Prints a summary table.
5. Writes three charts to `output/`.

### Options

```
python run_demo.py --customers 2000 --gamma 0.9 --base-reward 250 --seed 7
python run_demo.py --seed -1          # different random data every run
python run_demo.py --db other.db --output charts/
```

## Graph schema

```
nodes(node_id TEXT PK, label TEXT, properties JSON)
      label ∈ {Customer, Campaign, Channel, Conversion}

edges(edge_id INTEGER PK, source_id TEXT, target_id TEXT,
      relationship TEXT, timestamp TEXT (ISO-8601), properties JSON)
      relationship ∈ {VIEWED, CLICKED, CONVERTED_VIA}
```

Indexes cover `source_id`, `target_id` and `relationship`, plus a composite `(source_id, relationship, timestamp)` index for path walks.

```
(Customer) -[VIEWED|CLICKED @ts]-> (Campaign)         Campaign.properties.channel_id -> (Channel)
(Customer) -[CONVERTED_VIA @ts]->  (Conversion)
```

## How the engine works (`src/engine.py`)

All path traversal and latency maths runs in SQL:

```sql
WITH RECURSIVE
conversions AS (... edges WHERE relationship = 'CONVERTED_VIA'),
ordered_touches AS (
    SELECT ..., JULIANDAY(c.timestamp) - JULIANDAY(e.timestamp) AS delta_days,
           ROW_NUMBER() OVER (PARTITION BY c.conversion_id ORDER BY e.timestamp) AS step
    FROM edges e JOIN conversions c ON c.customer_id = e.source_id
    WHERE e.relationship IN ('CLICKED','VIEWED') AND e.timestamp <= c.timestamp),
journey AS (
    SELECT ... FROM ordered_touches WHERE step = 1             -- anchor: first touch
    UNION ALL
    SELECT ..., j.path || ' > ' || t.campaign_id               -- hop to the next touch
    FROM journey j JOIN ordered_touches t
      ON t.conversion_id = j.conversion_id AND t.step = j.step + 1)
SELECT campaign_id, SUM(:base_reward * POWER(:gamma, delta_days)) ...
```

`POWER` is registered as a Python UDF, so the code works on SQLite builds that were compiled without the math extension.

### Reading the summary table

| Column | Meaning |
|---|---|
| Total Conversions | The number of conversions whose path touched the campaign at least once (multi-touch reach). |
| Avg Latency (Days) | The mean `Δt` from the campaign's touches to their conversion. |
| Last-Touch Revenue | `Base_Reward ×` the number of conversions where the campaign was the final touch. |
| Markov Velocity-Adj. Revenue | `Σ Base_Reward × γ^Δt` over all of the campaign's touches on converting paths. |
| Share Shift | The campaign's share of Markov credit minus its share of Last-Touch credit. |

Each model's total differs. Last-Touch credits exactly one touch per conversion. The Markov model rewards every touch, and the reward of each touch decays with its age. Fast multi-touch journeys can therefore earn more than `Base_Reward` in total, and slow journeys earn much less. When you compare campaigns, use **Share Shift**, which normalises both models to 100%.

## Synthetic data (`src/generator.py`)

* **Funnel position.** Each campaign has a funnel stage. `LinkedIn_Video_Ad` and `Organic_Blog` tend to open journeys. `Email_Newsletter` tends to sit in the middle. `Retargeting_Banner` and `Google_Search_Branded` tend to close journeys.
* **Path length.** Journeys have 1 to 6 touches. Banners and video are mostly VIEWED; search is mostly CLICKED.
* **Conversion rate.** About 60% of customers convert. Their conversion latencies come from a mix of three profiles:
  * fast: 1–2 days (40%)
  * medium: 3–14 days (30%)
  * slow: 30–60 days (30%)
* **Reproducibility.** Output is the same for a given `--seed` (default 42).

## Outputs

| File | Content |
|---|---|
| `output/customer_journey_graph.png` | A NetworkX directed graph of the top 15 converting paths (Start → campaigns → Conversion), with edge width scaled to journey count. |
| `output/attribution_comparison.png` | Last-Touch vs. Markov revenue for each campaign. |
| `output/time_decay_curve.png` | The decay curve `100 × 0.85^Δt` over 30 days, with the half-life marked. |

## Project layout

```
├── README.md
├── requirements.txt
├── run_demo.py          # single entry point
└── src/
    ├── __init__.py
    ├── database.py      # SQLite graph schema + read/write helpers
    ├── generator.py     # synthetic multi-touch journeys
    ├── engine.py        # recursive-CTE path walks, Last-Touch & Markov models
    └── visualizer.py    # NetworkX / Matplotlib charts
```

## Using the engine directly

```python
from src.database import GraphDatabase
from src.engine import AttributionEngine, AttributionConfig

with GraphDatabase("marketing_graph.db") as db:
    engine = AttributionEngine(db, AttributionConfig(gamma=0.9))
    print(engine.compare_models())
    print(engine.touchpoints().head())   # every touch with delta_days and its reward
    print(engine.top_paths(5))
```
