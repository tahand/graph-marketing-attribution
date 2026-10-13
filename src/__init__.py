"""Multi-Touch Marketing Attribution Engine with Discounted Markov Rewards.

A zero-cost knowledge graph stored in local SQLite. Customer journeys are
modelled as nodes (Customer, Campaign, Channel, Conversion) and edges
(VIEWED, CLICKED, CONVERTED_VIA), then attributed with two models:

* Last-Touch Attribution - 100% of credit to the final touch.
* Discounted Markov Attribution - every touch earns
  ``Base_Reward * gamma ** delta_t`` where ``delta_t`` is the latency in days
  between the touch and the conversion.
"""

__version__ = "1.0.0"
