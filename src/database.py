"""SQLite knowledge-graph storage layer.

The graph is stored in two tables:

``nodes``
    ``node_id`` (TEXT PK), ``label`` (Customer | Campaign | Channel | Conversion),
    ``properties`` (JSON text).

``edges``
    ``edge_id`` (INTEGER PK), ``source_id``, ``target_id``,
    ``relationship`` (CLICKED | VIEWED | CONVERTED_VIA), ``timestamp`` (ISO-8601),
    ``properties`` (JSON text).
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping

DEFAULT_DB_PATH = Path("marketing_graph.db")

NODE_LABELS = ("Customer", "Campaign", "Channel", "Conversion")
RELATIONSHIPS = ("CLICKED", "VIEWED", "CONVERTED_VIA")

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS nodes (
    node_id    TEXT PRIMARY KEY,
    label      TEXT NOT NULL CHECK (label IN ({", ".join(f"'{l}'" for l in NODE_LABELS)})),
    properties JSON NOT NULL DEFAULT '{{}}'
);

CREATE TABLE IF NOT EXISTS edges (
    edge_id      INTEGER PRIMARY KEY,
    source_id    TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
    target_id    TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
    relationship TEXT NOT NULL CHECK (relationship IN ({", ".join(f"'{r}'" for r in RELATIONSHIPS)})),
    timestamp    TEXT NOT NULL,
    properties   JSON NOT NULL DEFAULT '{{}}'
);

CREATE INDEX IF NOT EXISTS idx_nodes_label         ON nodes(label);
CREATE INDEX IF NOT EXISTS idx_edges_source        ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target        ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_relationship  ON edges(relationship);
CREATE INDEX IF NOT EXISTS idx_edges_source_rel_ts ON edges(source_id, relationship, timestamp);
"""


class DatabaseError(RuntimeError):
    """Raised when the graph database cannot be created or queried."""


def _safe_pow(base: float | None, exponent: float | None) -> float | None:
    """SQL ``POWER(x, y)`` implementation that works on any SQLite build.

    Not every SQLite shipped with Python is compiled with the math extension,
    so the engine registers this function explicitly.
    """
    if base is None or exponent is None:
        return None
    try:
        return math.pow(base, exponent)
    except (OverflowError, ValueError):
        return None


class GraphDatabase:
    """Thin wrapper around a SQLite connection holding the marketing graph.

    Usable as a context manager::

        with GraphDatabase("marketing_graph.db") as db:
            db.add_nodes([...])
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------ #
    # Connection management
    # ------------------------------------------------------------------ #
    @property
    def conn(self) -> sqlite3.Connection:
        """Return an open connection, opening it lazily on first use."""
        if self._conn is None:
            try:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                self._conn = sqlite3.connect(self.db_path)
            except (sqlite3.Error, OSError) as exc:
                raise DatabaseError(f"Could not open database '{self.db_path}': {exc}") from exc
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.create_function("POWER", 2, _safe_pow, deterministic=True)
        return self._conn

    def close(self) -> None:
        """Close the underlying connection if it is open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "GraphDatabase":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Delete the database file (if any) and create a fresh schema."""
        self.close()
        try:
            if self.db_path.exists():
                self.db_path.unlink()
        except OSError as exc:
            raise DatabaseError(
                f"Could not delete '{self.db_path}'. Is it open in another program? ({exc})"
            ) from exc
        self.initialize()

    def initialize(self) -> None:
        """Create tables and indexes if they do not yet exist."""
        try:
            with self.conn:
                self.conn.executescript(SCHEMA_SQL)
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to initialise schema: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def add_nodes(self, nodes: Iterable[tuple[str, str, Mapping[str, Any]]]) -> int:
        """Insert or replace nodes given as ``(node_id, label, properties)``.

        Returns the number of rows written.
        """
        rows = [(nid, label, json.dumps(dict(props))) for nid, label, props in nodes]
        try:
            with self.conn:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO nodes (node_id, label, properties) VALUES (?, ?, ?)",
                    rows,
                )
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to insert nodes: {exc}") from exc
        return len(rows)

    def add_edges(
        self, edges: Iterable[tuple[str, str, str, str, Mapping[str, Any]]]
    ) -> int:
        """Insert edges given as ``(source_id, target_id, relationship, timestamp, properties)``.

        Returns the number of rows written.
        """
        rows = [(s, t, rel, ts, json.dumps(dict(props))) for s, t, rel, ts, props in edges]
        try:
            with self.conn:
                self.conn.executemany(
                    "INSERT INTO edges (source_id, target_id, relationship, timestamp, properties) "
                    "VALUES (?, ?, ?, ?, ?)",
                    rows,
                )
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to insert edges: {exc}") from exc
        return len(rows)

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def query(self, sql: str, params: Iterable[Any] | Mapping[str, Any] = ()) -> list[sqlite3.Row]:
        """Run a read query and return all rows."""
        try:
            return self.conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Query failed: {exc}\nSQL: {sql.strip()[:500]}") from exc

    def counts(self) -> dict[str, int]:
        """Return node counts per label and edge counts per relationship."""
        result: dict[str, int] = {}
        for row in self.query("SELECT label, COUNT(*) AS n FROM nodes GROUP BY label"):
            result[row["label"]] = row["n"]
        for row in self.query("SELECT relationship, COUNT(*) AS n FROM edges GROUP BY relationship"):
            result[row["relationship"]] = row["n"]
        return result
