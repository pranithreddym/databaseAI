"""
Transaction Isolation Levels — Dirty Read, Non-Repeatable Read, Phantom Read
==============================================================================

DB Architect notes:
  A transaction is a unit of work that must be Atomic, Consistent, Isolated,
  and Durable (ACID).  "Isolated" means concurrent transactions behave as if
  they ran serially — but full isolation is expensive.  The SQL standard
  therefore defines four levels, each trading some isolation for throughput:

  READ UNCOMMITTED — transactions can read uncommitted ("dirty") data from
    other in-flight transactions.  Maximally concurrent, minimally safe.
    Almost never used in practice because dirty reads produce logically
    impossible intermediate states.

  READ COMMITTED — a transaction only sees data committed before each
    individual statement executes.  Two reads of the same row within one
    transaction may return different values if another transaction commits
    between them (non-repeatable read).  PostgreSQL default; Oracle default.

  REPEATABLE READ — once a transaction reads a row it is guaranteed to see
    the same value for that row for the lifetime of the transaction.  However,
    new rows inserted by concurrent transactions may appear in range/aggregate
    queries (phantom read).  MySQL InnoDB default.  PostgreSQL implements this
    as full snapshot isolation (which also prevents phantoms).

  SERIALIZABLE — full isolation; result is identical to some serial execution
    of all concurrent transactions.  Prevents dirty reads, non-repeatable
    reads, AND phantom reads.  Highest isolation, highest contention cost.

  SQLite's implementation:
    • WAL mode gives readers a point-in-time snapshot for the duration of
      their transaction — equivalent to REPEATABLE READ / snapshot isolation.
    • Dirty reads are physically impossible: WAL readers never access
      uncommitted pages from other connections.
    • BEGIN EXCLUSIVE serialises all concurrent access: no other connection
      can read or write until the exclusive transaction commits, providing
      true SERIALIZABLE behaviour.

  This module uses threading.Event barriers to produce deterministic
  interleavings that expose (or prevent) each anomaly class.

Production parallels:
  - Payment processing: a funds-transfer must be SERIALIZABLE to prevent
    the "lost update" anomaly where two concurrent withdrawals both read the
    same balance, resulting in only one deduction being persisted.
  - Booking systems (airlines, hotels): phantom reads cause double-booking
    when two agents see "one seat remaining" simultaneously; SERIALIZABLE or
    optimistic locking prevents this.
  - Recommendation model A/B test metrics: READ COMMITTED is sufficient for
    dashboard reads; SERIALIZABLE is needed when atomically swapping model
    weights and clearing cached scores.
  - PostgreSQL advisory locks: an application-level SERIALIZABLE primitive
    used by Stripe and GitHub to prevent duplicate payment processing.
"""

import os
import sqlite3
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional


# SQL standard isolation level names (for documentation and output labels)
LEVEL_READ_UNCOMMITTED = "READ UNCOMMITTED"
LEVEL_READ_COMMITTED   = "READ COMMITTED"
LEVEL_REPEATABLE_READ  = "REPEATABLE READ"
LEVEL_SERIALIZABLE     = "SERIALIZABLE"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts_tx (
    account_id TEXT PRIMARY KEY,
    owner_id   TEXT NOT NULL,
    balance    REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS movies_tx (
    id     TEXT PRIMARY KEY,
    title  TEXT NOT NULL,
    genre  TEXT,
    rating REAL DEFAULT 0.0
);
"""

# Summary of which anomalies each level allows (True = anomaly CAN occur)
ISOLATION_MATRIX: List[Dict[str, Any]] = [
    {
        "level":              LEVEL_READ_UNCOMMITTED,
        "dirty_read":         True,
        "nonrepeatable_read": True,
        "phantom_read":       True,
        "sqlite_mechanism":   "Simulated (SQLite always prevents dirty reads)",
    },
    {
        "level":              LEVEL_READ_COMMITTED,
        "dirty_read":         False,
        "nonrepeatable_read": True,
        "phantom_read":       True,
        "sqlite_mechanism":   "Autocommit (isolation_level=None); each statement = own txn",
    },
    {
        "level":              LEVEL_REPEATABLE_READ,
        "dirty_read":         False,
        "nonrepeatable_read": False,
        "phantom_read":       False,  # SQLite WAL snapshot also prevents phantoms
        "sqlite_mechanism":   "BEGIN (snapshot isolation via WAL mode)",
    },
    {
        "level":              LEVEL_SERIALIZABLE,
        "dirty_read":         False,
        "nonrepeatable_read": False,
        "phantom_read":       False,
        "sqlite_mechanism":   "BEGIN EXCLUSIVE (blocks all concurrent readers/writers)",
    },
]


class TransactionIsolationDemo:
    """
    Demonstrates all four SQL isolation levels and their anomalies using
    SQLite with WAL mode and Python threading.

    Each public demonstrate_* method uses threading.Event barriers to produce
    a controlled, deterministic interleaving so that the anomaly (or its
    prevention) is reliably observable.

    A file-based SQLite database is used instead of :memory: because WAL mode
    (required for snapshot isolation) does not apply to in-memory databases,
    and because multi-connection concurrency requires a shared on-disk file.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            fd, self._db_path = tempfile.mkstemp(suffix=".db", prefix="txdemo_")
            os.close(fd)
            self._owns_file = True
        else:
            self._db_path = db_path
            self._owns_file = False

        conn = self._connect()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _connect(self, isolation_level: str = "", timeout: float = 5.0) -> sqlite3.Connection:
        """
        Open a new connection to the demo database.

        isolation_level=None  → autocommit; each statement is its own transaction
                                 (models READ COMMITTED behaviour)
        isolation_level=""    → Python's default deferred transaction management
        isolation_level=None + explicit BEGIN → snapshot transaction (WAL mode
                                 gives REPEATABLE READ / snapshot isolation)
        """
        conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            isolation_level=isolation_level,
            timeout=timeout,
        )
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Data setup
    # ------------------------------------------------------------------

    def seed(self, movies: list, users: list) -> None:
        """Seed movies and create one bank account per user (balance = 1 000.0)."""
        conn = self._connect()
        conn.executemany(
            "INSERT OR IGNORE INTO movies_tx (id, title, genre) "
            "VALUES (:id, :title, :genre)",
            movies,
        )
        for user in users:
            conn.execute(
                "INSERT OR IGNORE INTO accounts_tx "
                "(account_id, owner_id, balance) VALUES (?, ?, ?)",
                (f"acc_{user['id']}", user["id"], 1000.0),
            )
        conn.commit()
        conn.close()

    def reset_balance(self, account_id: str, balance: float) -> None:
        """Helper to reset an account balance between scenario runs."""
        conn = self._connect()
        conn.execute(
            "UPDATE accounts_tx SET balance = ? WHERE account_id = ?",
            (balance, account_id),
        )
        conn.commit()
        conn.close()

    def get_balance(self, account_id: str) -> Optional[float]:
        """Return the current committed balance for an account."""
        conn = self._connect(isolation_level=None)
        row = conn.execute(
            "SELECT balance FROM accounts_tx WHERE account_id = ?",
            (account_id,),
        ).fetchone()
        conn.close()
        return row[0] if row else None

    def close(self) -> None:
        if self._owns_file:
            for ext in ("", "-wal", "-shm"):
                try:
                    os.unlink(self._db_path + ext)
                except FileNotFoundError:
                    pass

    # ------------------------------------------------------------------
    # Anomaly 1: Dirty Read
    # ------------------------------------------------------------------

    def demonstrate_dirty_read(self) -> Dict[str, Any]:
        """
        Dirty Read: T2 reads a row inserted by T1 before T1 commits.

        SQLite prevents this at every isolation level: readers always obtain
        a snapshot of the last committed state and never see in-flight writes
        from other connections.  This method confirms the prevention.
        """
        results: Dict[str, Any] = {
            "scenario":               "Dirty Read",
            "t1_wrote_uncommitted":   False,
            "t2_saw_uncommitted_row": False,
            "anomaly_occurred":       False,
            "prevention":             "SQLite WAL snapshot — readers never see uncommitted pages",
        }

        ev_written    = threading.Event()
        ev_can_finish = threading.Event()

        def writer_t1() -> None:
            conn = self._connect(isolation_level=None)
            conn.execute("BEGIN")
            conn.execute(
                "INSERT INTO movies_tx (id, title, genre) "
                "VALUES ('dirty_tmp', 'Uncommitted Draft', 'test')"
            )
            results["t1_wrote_uncommitted"] = True
            ev_written.set()       # T2 may now attempt its read
            ev_can_finish.wait()   # hold the uncommitted write until T2 is done
            conn.execute("ROLLBACK")
            conn.close()

        def reader_t2() -> None:
            ev_written.wait()      # wait until T1 has written but NOT committed
            conn = self._connect(isolation_level=None)
            row = conn.execute(
                "SELECT id FROM movies_tx WHERE id = 'dirty_tmp'"
            ).fetchone()
            results["t2_saw_uncommitted_row"] = row is not None
            ev_can_finish.set()
            conn.close()

        t1 = threading.Thread(target=writer_t1, daemon=True)
        t2 = threading.Thread(target=reader_t2, daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        results["anomaly_occurred"] = results["t2_saw_uncommitted_row"]
        return results

    # ------------------------------------------------------------------
    # Anomaly 2: Non-Repeatable Read
    # ------------------------------------------------------------------

    def demonstrate_nonrepeatable_read(self) -> Dict[str, Any]:
        """
        Non-Repeatable Read: T1 reads the same row twice within a "session".
        Between the two reads T2 updates and commits that row.  With autocommit
        (READ COMMITTED) each SELECT is a new transaction, so T1's second read
        returns the updated value — a non-repeatable read.
        """
        results: Dict[str, Any] = {
            "scenario":         "Non-Repeatable Read",
            "t1_first_balance": None,
            "t1_second_balance": None,
            "anomaly_occurred": False,
        }

        account = "nrr_acc"
        setup = self._connect()
        setup.execute(
            "INSERT OR REPLACE INTO accounts_tx (account_id, owner_id, balance) "
            "VALUES (?, 'u01', 1000.0)",
            (account,),
        )
        setup.commit()
        setup.close()

        ev_first_read  = threading.Event()
        ev_update_done = threading.Event()

        def autocommit_reader_t1() -> None:
            conn = self._connect(isolation_level=None)  # READ COMMITTED: no txn wrapper
            row = conn.execute(
                "SELECT balance FROM accounts_tx WHERE account_id = ?", (account,)
            ).fetchone()
            results["t1_first_balance"] = row[0] if row else None
            ev_first_read.set()
            ev_update_done.wait()
            # Second SELECT — new autocommit transaction, sees T2's committed change
            row = conn.execute(
                "SELECT balance FROM accounts_tx WHERE account_id = ?", (account,)
            ).fetchone()
            results["t1_second_balance"] = row[0] if row else None
            conn.close()

        def writer_t2() -> None:
            ev_first_read.wait()
            conn = self._connect()
            conn.execute(
                "UPDATE accounts_tx SET balance = 250.0 WHERE account_id = ?",
                (account,),
            )
            conn.commit()
            ev_update_done.set()
            conn.close()

        t1 = threading.Thread(target=autocommit_reader_t1, daemon=True)
        t2 = threading.Thread(target=writer_t2, daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        results["anomaly_occurred"] = (
            results["t1_first_balance"] != results["t1_second_balance"]
        )
        return results

    # ------------------------------------------------------------------
    # Prevention 2: Repeatable Read via snapshot
    # ------------------------------------------------------------------

    def demonstrate_repeatable_read_prevention(self) -> Dict[str, Any]:
        """
        Repeatable Read Prevention: T1 wraps both reads in an explicit
        BEGIN/COMMIT block.  SQLite's WAL snapshot isolation gives T1 a
        stable view of the database from the moment of its first read —
        T2's committed update is invisible to T1's second read.
        """
        results: Dict[str, Any] = {
            "scenario":          "Repeatable Read Prevention",
            "t1_first_balance":  None,
            "t1_second_balance": None,
            "consistent":        False,
        }

        account = "rr_acc"
        setup = self._connect()
        setup.execute(
            "INSERT OR REPLACE INTO accounts_tx (account_id, owner_id, balance) "
            "VALUES (?, 'u01', 1000.0)",
            (account,),
        )
        setup.commit()
        setup.close()

        ev_first_read  = threading.Event()
        ev_update_done = threading.Event()

        def snapshot_reader_t1() -> None:
            conn = self._connect(isolation_level=None)
            conn.execute("BEGIN")   # snapshot taken on first read in WAL mode
            row = conn.execute(
                "SELECT balance FROM accounts_tx WHERE account_id = ?", (account,)
            ).fetchone()
            results["t1_first_balance"] = row[0] if row else None
            ev_first_read.set()
            ev_update_done.wait()
            # Second read — still within the same snapshot transaction
            row = conn.execute(
                "SELECT balance FROM accounts_tx WHERE account_id = ?", (account,)
            ).fetchone()
            results["t1_second_balance"] = row[0] if row else None
            conn.execute("COMMIT")
            conn.close()

        def concurrent_writer_t2() -> None:
            ev_first_read.wait()
            conn = self._connect()
            conn.execute(
                "UPDATE accounts_tx SET balance = 250.0 WHERE account_id = ?",
                (account,),
            )
            conn.commit()
            ev_update_done.set()
            conn.close()

        t1 = threading.Thread(target=snapshot_reader_t1, daemon=True)
        t2 = threading.Thread(target=concurrent_writer_t2, daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        results["consistent"] = (
            results["t1_first_balance"] == results["t1_second_balance"]
        )
        return results

    # ------------------------------------------------------------------
    # Anomaly 3: Phantom Read
    # ------------------------------------------------------------------

    def demonstrate_phantom_read(self) -> Dict[str, Any]:
        """
        Phantom Read: T1 executes the same aggregate query twice.  Between
        runs T2 inserts a new row matching the predicate.  In autocommit
        (READ COMMITTED) mode each COUNT is a separate transaction and the
        result changes — a phantom row has appeared.
        """
        results: Dict[str, Any] = {
            "scenario":       "Phantom Read",
            "t1_first_count": None,
            "t1_second_count": None,
            "anomaly_occurred": False,
        }

        # Ensure no leftover phantom rows from previous runs
        cleanup = self._connect()
        cleanup.execute("DELETE FROM movies_tx WHERE id LIKE 'phantom_%'")
        cleanup.commit()
        cleanup.close()

        ev_first_count  = threading.Event()
        ev_insert_done  = threading.Event()

        def autocommit_counter_t1() -> None:
            conn = self._connect(isolation_level=None)  # READ COMMITTED
            count = conn.execute(
                "SELECT COUNT(*) FROM movies_tx WHERE genre = 'sci-fi'"
            ).fetchone()[0]
            results["t1_first_count"] = count
            ev_first_count.set()
            ev_insert_done.wait()
            # Second count — new autocommit transaction, sees T2's inserted row
            count = conn.execute(
                "SELECT COUNT(*) FROM movies_tx WHERE genre = 'sci-fi'"
            ).fetchone()[0]
            results["t1_second_count"] = count
            conn.close()

        def inserter_t2() -> None:
            ev_first_count.wait()
            conn = self._connect()
            conn.execute(
                "INSERT INTO movies_tx (id, title, genre) "
                "VALUES ('phantom_1', 'The Phantom Galaxy', 'sci-fi')"
            )
            conn.commit()
            ev_insert_done.set()
            conn.close()

        t1 = threading.Thread(target=autocommit_counter_t1, daemon=True)
        t2 = threading.Thread(target=inserter_t2, daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        results["anomaly_occurred"] = (
            results["t1_first_count"] != results["t1_second_count"]
        )
        return results

    # ------------------------------------------------------------------
    # Prevention 3: Phantom Prevention via snapshot
    # ------------------------------------------------------------------

    def demonstrate_phantom_prevention(self) -> Dict[str, Any]:
        """
        Phantom Prevention: T1 wraps both COUNT queries in an explicit
        transaction.  SQLite WAL snapshot isolation ensures T2's committed
        INSERT is invisible to T1's second COUNT — no phantom row appears.
        """
        results: Dict[str, Any] = {
            "scenario":          "Phantom Read Prevention",
            "t1_first_count":    None,
            "t1_second_count":   None,
            "consistent":        False,
        }

        cleanup = self._connect()
        cleanup.execute("DELETE FROM movies_tx WHERE id LIKE 'phantom2_%'")
        cleanup.commit()
        cleanup.close()

        ev_first_count  = threading.Event()
        ev_insert_done  = threading.Event()

        def snapshot_counter_t1() -> None:
            conn = self._connect(isolation_level=None)
            conn.execute("BEGIN")   # snapshot locked in
            count = conn.execute(
                "SELECT COUNT(*) FROM movies_tx WHERE genre = 'sci-fi'"
            ).fetchone()[0]
            results["t1_first_count"] = count
            ev_first_count.set()
            ev_insert_done.wait()
            count = conn.execute(
                "SELECT COUNT(*) FROM movies_tx WHERE genre = 'sci-fi'"
            ).fetchone()[0]
            results["t1_second_count"] = count
            conn.execute("COMMIT")
            conn.close()

        def inserter_t2() -> None:
            ev_first_count.wait()
            conn = self._connect()
            conn.execute(
                "INSERT INTO movies_tx (id, title, genre) "
                "VALUES ('phantom2_1', 'Phantom Sequel', 'sci-fi')"
            )
            conn.commit()
            ev_insert_done.set()
            conn.close()

        t1 = threading.Thread(target=snapshot_counter_t1, daemon=True)
        t2 = threading.Thread(target=inserter_t2, daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        results["consistent"] = (
            results["t1_first_count"] == results["t1_second_count"]
        )
        return results

    # ------------------------------------------------------------------
    # SERIALIZABLE: BEGIN EXCLUSIVE
    # ------------------------------------------------------------------

    def demonstrate_exclusive_lock(self) -> Dict[str, Any]:
        """
        SERIALIZABLE via BEGIN EXCLUSIVE: T2 cannot start its own EXCLUSIVE
        transaction until T1 commits.  Both connections compete for the
        write lock, serialising all concurrent modifications.

        Note on WAL mode: WAL allows readers to proceed concurrently with a
        writer holding an exclusive lock.  To demonstrate observable blocking,
        T2 also requests an EXCLUSIVE transaction so it must wait for T1 to
        release the write lock before it can begin.
        """
        results: Dict[str, Any] = {
            "scenario":          "Exclusive Lock (SERIALIZABLE)",
            "t1_lock_acquired":  False,
            "t2_blocked_ms":     None,
            "t2_was_blocked":    False,
            "t2_final_balance":  None,
        }

        account = "excl_acc"
        setup = self._connect()
        setup.execute(
            "INSERT OR REPLACE INTO accounts_tx (account_id, owner_id, balance) "
            "VALUES (?, 'u01', 1000.0)",
            (account,),
        )
        setup.commit()
        setup.close()

        ev_locked = threading.Event()

        def exclusive_writer_t1() -> None:
            conn = self._connect(isolation_level=None)
            conn.execute("BEGIN EXCLUSIVE")
            results["t1_lock_acquired"] = True
            ev_locked.set()
            time.sleep(0.12)    # hold exclusive lock for 120 ms
            conn.execute(
                "UPDATE accounts_tx SET balance = 800.0 WHERE account_id = ?",
                (account,),
            )
            conn.execute("COMMIT")
            conn.close()

        def blocked_writer_t2() -> None:
            """T2 also requests EXCLUSIVE — must wait for T1 to commit."""
            ev_locked.wait()    # ensure T1 holds the lock before T2 starts
            conn = self._connect(isolation_level=None, timeout=5.0)
            t_start = time.perf_counter()
            conn.execute("BEGIN EXCLUSIVE")   # blocks until T1 releases the lock
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            results["t2_blocked_ms"]  = round(elapsed_ms, 1)
            results["t2_was_blocked"] = elapsed_ms > 50   # blocked > 50 ms
            # T2 now reads the value T1 committed
            row = conn.execute(
                "SELECT balance FROM accounts_tx WHERE account_id = ?",
                (account,),
            ).fetchone()
            results["t2_final_balance"] = row[0] if row else None
            conn.execute("COMMIT")
            conn.close()

        t1 = threading.Thread(target=exclusive_writer_t1, daemon=True)
        t2 = threading.Thread(target=blocked_writer_t2, daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        return results

    # ------------------------------------------------------------------
    # Real-world scenario: Payment Lost Update
    # ------------------------------------------------------------------

    def simulate_payment_scenario(
        self,
        initial_balance: float = 500.0,
        withdrawal: float = 300.0,
    ) -> Dict[str, Any]:
        """
        Real-world parallel: two concurrent ATM withdrawals from the same account.

        Without serialisation both transactions read the same balance, compute
        the same new balance, and commit — one deduction is silently lost.
        With BEGIN EXCLUSIVE the second transaction reads the already-updated
        balance and correctly rejects the withdrawal.
        """
        account = "payment_acc"

        # --- Scenario A: unserialized (lost update) --------------------------
        setup = self._connect()
        setup.execute(
            "INSERT OR REPLACE INTO accounts_tx (account_id, owner_id, balance) "
            "VALUES (?, 'u01', ?)",
            (account, initial_balance),
        )
        setup.commit()
        setup.close()

        unser: Dict[str, Any] = {
            "t1_read": None, "t2_read": None,
            "t1_wrote": None, "t2_wrote": None,
            "final_balance": None,
            "withdrawals_applied": 0,
        }
        barrier = threading.Barrier(2)  # both threads read before either writes

        def unserialized_withdrawal(tid: int) -> None:
            conn = self._connect(isolation_level=None)
            row = conn.execute(
                "SELECT balance FROM accounts_tx WHERE account_id = ?", (account,)
            ).fetchone()
            balance = row[0]
            unser[f"t{tid}_read"] = balance
            barrier.wait()          # synchronise: both reads happen before any write
            new_bal = balance - withdrawal
            conn.execute(
                "UPDATE accounts_tx SET balance = ? WHERE account_id = ?",
                (new_bal, account),
            )
            unser[f"t{tid}_wrote"] = new_bal
            conn.close()

        t1 = threading.Thread(target=unserialized_withdrawal, args=(1,), daemon=True)
        t2 = threading.Thread(target=unserialized_withdrawal, args=(2,), daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        unser["final_balance"] = self.get_balance(account)
        # Both threads "succeeded" but balance only dropped by one withdrawal
        unser["withdrawals_applied"] = round(
            (initial_balance - unser["final_balance"]) / withdrawal
        )

        # --- Scenario B: SERIALIZABLE (BEGIN EXCLUSIVE) ----------------------
        self.reset_balance(account, initial_balance)

        ser: Dict[str, Any] = {
            "withdrawals_succeeded": 0,
            "withdrawals_rejected":  0,
            "final_balance": None,
        }
        ser_lock = threading.Lock()

        def serialized_withdrawal() -> None:
            conn = self._connect(isolation_level=None, timeout=5.0)
            try:
                conn.execute("BEGIN EXCLUSIVE")
                row = conn.execute(
                    "SELECT balance FROM accounts_tx WHERE account_id = ?", (account,)
                ).fetchone()
                balance = row[0]
                if balance >= withdrawal:
                    conn.execute(
                        "UPDATE accounts_tx SET balance = ? WHERE account_id = ?",
                        (balance - withdrawal, account),
                    )
                    conn.execute("COMMIT")
                    with ser_lock:
                        ser["withdrawals_succeeded"] += 1
                else:
                    conn.execute("ROLLBACK")
                    with ser_lock:
                        ser["withdrawals_rejected"] += 1
            except sqlite3.OperationalError:
                with ser_lock:
                    ser["withdrawals_rejected"] += 1
            finally:
                conn.close()

        t1 = threading.Thread(target=serialized_withdrawal, daemon=True)
        t2 = threading.Thread(target=serialized_withdrawal, daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        ser["final_balance"] = self.get_balance(account)

        return {"unserialized": unser, "serialized": ser}
