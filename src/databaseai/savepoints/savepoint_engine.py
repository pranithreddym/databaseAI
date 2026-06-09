"""
Savepoints and Nested Rollbacks — Partial Transaction Control
=============================================================

DB Architect notes:
  Every SQL transaction is all-or-nothing: COMMIT persists everything,
  ROLLBACK discards everything.  SAVEPOINTs break that binary by adding named
  checkpoints *inside* an active transaction.  Work can be rolled back to any
  savepoint — undoing only what happened since that point — while the
  surrounding transaction stays open.

  SQLite savepoint syntax:
    SAVEPOINT sp_name          — create a named checkpoint
    ROLLBACK TO SAVEPOINT sp   — undo all work since sp; outer txn stays alive
    RELEASE SAVEPOINT sp       — merge sp's work into the enclosing transaction
                                 (or outer savepoint if nested); removes sp

  Nesting:
    Savepoints stack like a call frame.  Each new SAVEPOINT adds a layer;
    RELEASE pops the top layer; ROLLBACK TO rewinds to that layer and
    destroys every inner savepoint above it.  This is exactly how Django's
    transaction.atomic() implements nested transactions — each inner block
    opens a savepoint, its success issues RELEASE, its failure issues
    ROLLBACK TO.

  Why prefer ROLLBACK TO SAVEPOINT over a full ROLLBACK:
    A full ROLLBACK aborts the ENTIRE transaction.  In a multi-step
    pipeline — order creation, slot reservation, payment charge, subscription
    activation — a transient failure in step 3 should not invalidate the
    audit records already written by steps 1 and 2.  Savepoints let each
    step fail and retry independently while the outer transaction accumulates
    committed work.

  Connection-level note:
    SQLite does not expose savepoints through Python's isolation_level
    parameter.  You must issue SAVEPOINT / RELEASE / ROLLBACK TO as raw SQL
    execute() calls on a connection that has an open transaction
    (isolation_level=None with an explicit BEGIN).

Production parallels:
  - Django ORM: every nested transaction.atomic() block opens a SAVEPOINT;
    an unhandled exception in the inner block triggers ROLLBACK TO; a
    successful exit triggers RELEASE.  This gives per-request partial rollback
    without opening a second database connection.
  - PostgreSQL stored procedures (PL/pgSQL): the EXCEPTION clause uses an
    implicit SAVEPOINT / ROLLBACK TO under the hood, allowing a stored
    procedure to catch and handle errors while keeping the outer transaction
    alive.
  - Stripe payment pipeline: each stage (fraud check, balance check, card
    charge, ledger debit) is wrapped in a savepoint so a transient gateway
    timeout in one stage can be retried without discarding the audit trail
    written by earlier stages.
  - Batch ETL loaders: per-row savepoints allow a bulk import to skip
    malformed records and continue processing the rest of the batch —
    the canonical "skip-bad-rows" pattern used in Airflow DAG operators and
    Spark JDBC writers with error tolerances.
"""

import os
import sqlite3
import tempfile
from typing import Any, Dict, List, Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS watchlist (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id   TEXT NOT NULL,
    movie_id  TEXT NOT NULL,
    added_at  TEXT DEFAULT (datetime('now')),
    UNIQUE(user_id, movie_id)
);

CREATE TABLE IF NOT EXISTS purchase_orders (
    order_id    TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    plan        TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS subscriptions (
    sub_id       TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    plan         TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'inactive',
    activated_at TEXT
);

CREATE TABLE IF NOT EXISTS ingest_catalog (
    movie_id  TEXT PRIMARY KEY,
    title     TEXT NOT NULL,
    genre     TEXT,
    year      INTEGER,
    rating    REAL
);
"""


class SavepointDemo:
    """
    Demonstrates SQLite SAVEPOINT / RELEASE / ROLLBACK TO semantics for
    partial transaction rollbacks and fault-tolerant batch processing.

    Uses a file-based database so that multiple save/rollback cycles are
    independently verifiable.  Each demonstration method cleans its own
    rows before operating so repeated calls within a test suite are safe.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            fd, self._db_path = tempfile.mkstemp(suffix=".db", prefix="spdemo_")
            os.close(fd)
            self._owns_file = True
        else:
            self._db_path = db_path
            self._owns_file = False

        conn = self._connect()
        conn.executescript(_SCHEMA)
        conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self._db_path,
            isolation_level=None,   # manual transaction control via raw SQL
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        return conn

    def seed(self, movies: list) -> None:
        """Populate ingest_catalog from the seed movie list."""
        conn = self._connect()
        conn.execute("BEGIN")
        conn.executemany(
            "INSERT OR IGNORE INTO ingest_catalog (movie_id, title, genre, year) "
            "VALUES (:id, :title, :genre, :year)",
            movies,
        )
        conn.execute("COMMIT")
        conn.close()

    def row_count(self, table: str) -> int:
        conn = self._connect()
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        conn.close()
        return n

    def close(self) -> None:
        if self._owns_file:
            for ext in ("", "-wal", "-shm"):
                try:
                    os.unlink(self._db_path + ext)
                except FileNotFoundError:
                    pass

    # ------------------------------------------------------------------
    # Demo 1: Basic SAVEPOINT — partial rollback within a transaction
    # ------------------------------------------------------------------

    def basic_savepoint(self, user_id: str = "u01") -> Dict[str, Any]:
        """
        Insert item A, create savepoint, insert item B, ROLLBACK TO savepoint
        (removes B, A and outer transaction survive), RELEASE, COMMIT.

        Shows that ROLLBACK TO is a partial undo, not a transaction abort.
        """
        conn = self._connect()
        conn.execute("BEGIN")
        conn.execute("DELETE FROM watchlist WHERE user_id = ?", (user_id,))
        conn.execute("COMMIT")
        conn.close()

        results: Dict[str, Any] = {}

        conn = self._connect()
        conn.execute("BEGIN")

        conn.execute(
            "INSERT INTO watchlist (user_id, movie_id) VALUES (?, 'm01')", (user_id,)
        )
        results["count_after_insert_a"] = conn.execute(
            "SELECT COUNT(*) FROM watchlist WHERE user_id = ?", (user_id,)
        ).fetchone()[0]

        conn.execute("SAVEPOINT sp_after_a")

        conn.execute(
            "INSERT INTO watchlist (user_id, movie_id) VALUES (?, 'm02')", (user_id,)
        )
        results["count_after_insert_b"] = conn.execute(
            "SELECT COUNT(*) FROM watchlist WHERE user_id = ?", (user_id,)
        ).fetchone()[0]

        # ROLLBACK TO — removes m02, outer transaction stays alive
        conn.execute("ROLLBACK TO SAVEPOINT sp_after_a")
        results["count_after_rollback"] = conn.execute(
            "SELECT COUNT(*) FROM watchlist WHERE user_id = ?", (user_id,)
        ).fetchone()[0]

        conn.execute("RELEASE SAVEPOINT sp_after_a")
        conn.execute("COMMIT")
        conn.close()

        conn2 = self._connect()
        rows = conn2.execute(
            "SELECT movie_id FROM watchlist WHERE user_id = ?", (user_id,)
        ).fetchall()
        conn2.close()
        movie_ids = {r["movie_id"] for r in rows}

        results["item_a_committed"] = "m01" in movie_ids
        results["item_b_committed"] = "m02" in movie_ids
        results["final_count"] = len(movie_ids)
        return results

    # ------------------------------------------------------------------
    # Demo 2: Nested SAVEPOINTs — outer rollback discards inner
    # ------------------------------------------------------------------

    def nested_savepoints(self, user_id: str = "u02") -> Dict[str, Any]:
        """
        Demonstrates savepoint stacking.  Rolling back to an outer savepoint
        discards all inner savepoints and their work.

        Stack:
          BEGIN
            INSERT m01
            SAVEPOINT outer_sp
              INSERT m02
              SAVEPOINT inner_sp
                INSERT m03
              -- ROLLBACK TO outer_sp  <- discards inner_sp AND m02 AND m03
            RELEASE outer_sp
          COMMIT                        <- only m01 persists
        """
        conn = self._connect()
        conn.execute("BEGIN")
        conn.execute("DELETE FROM watchlist WHERE user_id = ?", (user_id,))
        conn.execute("COMMIT")
        conn.close()

        results: Dict[str, Any] = {}

        conn = self._connect()
        conn.execute("BEGIN")

        conn.execute(
            "INSERT INTO watchlist (user_id, movie_id) VALUES (?, 'm01')", (user_id,)
        )
        results["count_after_m01"] = conn.execute(
            "SELECT COUNT(*) FROM watchlist WHERE user_id = ?", (user_id,)
        ).fetchone()[0]

        conn.execute("SAVEPOINT outer_sp")

        conn.execute(
            "INSERT INTO watchlist (user_id, movie_id) VALUES (?, 'm02')", (user_id,)
        )
        conn.execute("SAVEPOINT inner_sp")
        conn.execute(
            "INSERT INTO watchlist (user_id, movie_id) VALUES (?, 'm03')", (user_id,)
        )

        results["count_with_all_three"] = conn.execute(
            "SELECT COUNT(*) FROM watchlist WHERE user_id = ?", (user_id,)
        ).fetchone()[0]

        # ROLLBACK TO outer_sp discards inner_sp, m02, and m03 simultaneously
        conn.execute("ROLLBACK TO SAVEPOINT outer_sp")
        results["count_after_outer_rollback"] = conn.execute(
            "SELECT COUNT(*) FROM watchlist WHERE user_id = ?", (user_id,)
        ).fetchone()[0]

        conn.execute("RELEASE SAVEPOINT outer_sp")
        conn.execute("COMMIT")
        conn.close()

        conn2 = self._connect()
        rows = conn2.execute(
            "SELECT movie_id FROM watchlist WHERE user_id = ?", (user_id,)
        ).fetchall()
        conn2.close()
        movie_ids = {r["movie_id"] for r in rows}

        results["survived_movies"] = sorted(movie_ids)
        results["only_m01_survived"] = (movie_ids == {"m01"})
        return results

    # ------------------------------------------------------------------
    # Demo 3: Multi-step purchase flow with per-step savepoints and retry
    # ------------------------------------------------------------------

    def purchase_flow(
        self,
        order_id: str = "ord_001",
        user_id: str = "u01",
        plan: str = "premium",
        fail_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Four-step subscription purchase flow, each step wrapped in a savepoint.

        Steps: create_order -> reserve_slots -> charge_payment -> activate_sub

        fail_at: if given, that step raises a simulated error on its first
        attempt.  The step is rolled back via ROLLBACK TO its savepoint, then
        retried (simulating a transient failure).  The retry succeeds; earlier
        steps are never rewound.

        This mirrors Django's @transaction.atomic decorator on individual
        service methods: the method's savepoint is rolled back on error, the
        caller's outer transaction is unaffected.
        """
        conn = self._connect()
        conn.execute("BEGIN")
        conn.execute("DELETE FROM purchase_orders WHERE order_id = ?", (order_id,))
        conn.execute("DELETE FROM subscriptions WHERE user_id = ?", (user_id,))
        conn.execute("COMMIT")
        conn.close()

        steps: List[Dict[str, Any]] = []
        _failed_once: Dict[str, bool] = {}

        conn = self._connect()
        conn.execute("BEGIN")

        def run_step(name: str, sql_fn) -> None:
            """Execute sql_fn inside a savepoint; retry once on failure."""
            conn.execute(f"SAVEPOINT sp_{name}")
            try:
                if fail_at == name and not _failed_once.get(name):
                    _failed_once[name] = True
                    raise RuntimeError(f"Transient error at step '{name}'")
                sql_fn()
                conn.execute(f"RELEASE SAVEPOINT sp_{name}")
                steps.append({"step": name, "outcome": "success"})
            except RuntimeError as exc:
                conn.execute(f"ROLLBACK TO SAVEPOINT sp_{name}")
                steps.append({"step": name, "outcome": "retrying", "error": str(exc)})
                # Retry: fail_at check skipped because _failed_once[name] is now True
                sql_fn()
                conn.execute(f"RELEASE SAVEPOINT sp_{name}")
                steps[-1]["outcome"] = "success_after_retry"

        def step_create_order() -> None:
            conn.execute(
                "INSERT INTO purchase_orders (order_id, user_id, status, plan) "
                "VALUES (?, ?, 'created', ?)",
                (order_id, user_id, plan),
            )

        def step_reserve_slots() -> None:
            conn.execute(
                "UPDATE purchase_orders SET status = 'slots_reserved' "
                "WHERE order_id = ?",
                (order_id,),
            )

        def step_charge_payment() -> None:
            conn.execute(
                "UPDATE purchase_orders SET status = 'charged' WHERE order_id = ?",
                (order_id,),
            )

        def step_activate_sub() -> None:
            conn.execute(
                "INSERT INTO subscriptions "
                "(sub_id, user_id, plan, status, activated_at) "
                "VALUES (?, ?, ?, 'active', datetime('now'))",
                (f"sub_{order_id}", user_id, plan),
            )
            conn.execute(
                "UPDATE purchase_orders SET status = 'completed' WHERE order_id = ?",
                (order_id,),
            )

        run_step("create_order",   step_create_order)
        run_step("reserve_slots",  step_reserve_slots)
        run_step("charge_payment", step_charge_payment)
        run_step("activate_sub",   step_activate_sub)

        conn.execute("COMMIT")
        conn.close()

        conn2 = self._connect()
        order_row = conn2.execute(
            "SELECT * FROM purchase_orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        sub_row = conn2.execute(
            "SELECT * FROM subscriptions WHERE user_id = ?", (user_id,)
        ).fetchone()
        conn2.close()

        return {
            "steps": steps,
            "order_status": dict(order_row) if order_row else None,
            "subscription": dict(sub_row) if sub_row else None,
            "all_steps_succeeded": all(
                s["outcome"] in ("success", "success_after_retry") for s in steps
            ),
        }

    # ------------------------------------------------------------------
    # Demo 4: Batch ingestion with per-row savepoints
    # ------------------------------------------------------------------

    def batch_ingest(
        self,
        records: List[Dict[str, Any]],
        fail_ids: Optional[set] = None,
    ) -> Dict[str, Any]:
        """
        Insert a batch of movie records using a per-row savepoint strategy.

        Records whose IDs are in fail_ids raise a ValueError (simulating a
        validation error such as a missing required field).  Each bad row is
        skipped via ROLLBACK TO its savepoint; the batch continues and commits
        all good rows.

        Pattern:
          BEGIN
          for each record:
            SAVEPOINT sp_row_{id}
            INSERT ...
            if error:  ROLLBACK TO sp_row_{id}  -> skip this row
            else:      RELEASE sp_row_{id}
          COMMIT
        """
        fail_ids = fail_ids or set()
        succeeded: List[str] = []
        skipped: List[str] = []

        conn = self._connect()
        conn.execute("BEGIN")

        for rec in records:
            rid = rec["movie_id"]
            conn.execute(f"SAVEPOINT sp_row_{rid}")
            try:
                if rid in fail_ids:
                    raise ValueError(f"Validation failed for record '{rid}'")
                conn.execute(
                    "INSERT OR IGNORE INTO ingest_catalog "
                    "(movie_id, title, genre, year, rating) "
                    "VALUES (:movie_id, :title, :genre, :year, :rating)",
                    rec,
                )
                conn.execute(f"RELEASE SAVEPOINT sp_row_{rid}")
                succeeded.append(rid)
            except (ValueError, sqlite3.IntegrityError):
                conn.execute(f"ROLLBACK TO SAVEPOINT sp_row_{rid}")
                conn.execute(f"RELEASE SAVEPOINT sp_row_{rid}")
                skipped.append(rid)

        conn.execute("COMMIT")
        conn.close()

        return {
            "total": len(records),
            "succeeded": succeeded,
            "skipped": skipped,
            "success_count": len(succeeded),
            "skip_count": len(skipped),
        }

    # ------------------------------------------------------------------
    # Demo 5: ROLLBACK TO SAVEPOINT vs full ROLLBACK
    # ------------------------------------------------------------------

    def savepoint_vs_full_rollback(self, user_id: str = "u03") -> Dict[str, Any]:
        """
        Contrast ROLLBACK TO SAVEPOINT (partial undo, outer transaction lives)
        with a full ROLLBACK (entire transaction discarded).

        Scenario A — savepoint partial rollback:
          BEGIN -> insert m04 -> SAVEPOINT sp_mid -> insert m05 ->
          ROLLBACK TO sp_mid -> insert m06 -> RELEASE sp_mid -> COMMIT
          Persisted: {m04, m06}  (m05 was rolled back at the savepoint)

        Scenario B — full transaction rollback:
          BEGIN -> insert m04 -> insert m05 -> ROLLBACK
          Persisted: {}  (entire transaction was discarded)
        """
        conn = self._connect()
        conn.execute("BEGIN")
        conn.execute("DELETE FROM watchlist WHERE user_id = ?", (user_id,))
        conn.execute("COMMIT")
        conn.close()

        # --- Scenario A: savepoint partial rollback ---
        conn = self._connect()
        conn.execute("BEGIN")
        conn.execute(
            "INSERT INTO watchlist (user_id, movie_id) VALUES (?, 'm04')", (user_id,)
        )
        conn.execute("SAVEPOINT sp_mid")
        conn.execute(
            "INSERT INTO watchlist (user_id, movie_id) VALUES (?, 'm05')", (user_id,)
        )
        conn.execute("ROLLBACK TO SAVEPOINT sp_mid")  # removes m05
        conn.execute(
            "INSERT INTO watchlist (user_id, movie_id) VALUES (?, 'm06')", (user_id,)
        )
        conn.execute("RELEASE SAVEPOINT sp_mid")
        conn.execute("COMMIT")
        conn.close()

        conn2 = self._connect()
        rows_a = conn2.execute(
            "SELECT movie_id FROM watchlist WHERE user_id = ?", (user_id,)
        ).fetchall()
        conn2.close()
        scenario_a = sorted(r["movie_id"] for r in rows_a)

        # --- Scenario B: full rollback ---
        conn = self._connect()
        conn.execute("BEGIN")
        conn.execute("DELETE FROM watchlist WHERE user_id = ?", (user_id,))
        conn.execute("COMMIT")
        conn.close()

        conn = self._connect()
        conn.execute("BEGIN")
        conn.execute(
            "INSERT INTO watchlist (user_id, movie_id) VALUES (?, 'm04')", (user_id,)
        )
        conn.execute(
            "INSERT INTO watchlist (user_id, movie_id) VALUES (?, 'm05')", (user_id,)
        )
        conn.execute("ROLLBACK")   # discards the entire transaction
        conn.close()

        conn2 = self._connect()
        rows_b = conn2.execute(
            "SELECT movie_id FROM watchlist WHERE user_id = ?", (user_id,)
        ).fetchall()
        conn2.close()
        scenario_b = sorted(r["movie_id"] for r in rows_b)

        return {
            "scenario_a_persisted": scenario_a,
            "scenario_b_persisted": scenario_b,
            "scenario_a_count": len(scenario_a),
            "scenario_b_count": len(scenario_b),
        }
