"""
Demo 13: Transactions Deep-Dive
================================
Demonstrates all four SQL isolation levels (READ UNCOMMITTED, READ COMMITTED,
REPEATABLE READ, SERIALIZABLE) and the three concurrent-access anomalies each
level prevents: dirty reads, non-repeatable reads, and phantom reads.

Uses SQLite with WAL mode and Python threading to produce controlled,
deterministic interleavings so each anomaly is reliably observable.

Real-world parallel: payment processing isolation levels — the same isolation
hierarchy that protects ATM withdrawals, airline seat reservations, and
e-commerce inventory updates against race conditions.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from databaseai.seed_data import MOVIES, USERS
from databaseai.transactions import (
    TransactionIsolationDemo,
    LEVEL_READ_UNCOMMITTED,
    LEVEL_READ_COMMITTED,
    LEVEL_REPEATABLE_READ,
    LEVEL_SERIALIZABLE,
    ISOLATION_MATRIX,
)

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bool_cell(flag: bool, true_label: str = "YES", false_label: str = "NO") -> str:
    if flag:
        return f"[red]{true_label}[/red]"
    return f"[green]{false_label}[/green]"


def _check(ok: bool) -> str:
    return "[bold green]✓ PREVENTED[/bold green]" if ok else "[bold red]✗ OCCURRED[/bold red]"


def _label(anomaly_occurred: bool) -> str:
    return "[red]✗ anomaly[/red]" if anomaly_occurred else "[green]✓ prevented[/green]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    console.rule("[bold cyan]Transactions Deep-Dive[/bold cyan]")
    console.print(
        "[dim]Real-world parallel: payment processing isolation levels —\n"
        "  the same trade-offs that govern ATM withdrawals, airline bookings,\n"
        "  and e-commerce inventory updates.[/dim]\n"
    )

    demo = TransactionIsolationDemo()
    demo.seed(MOVIES, USERS)

    # -----------------------------------------------------------------------
    # Section 1: Isolation Level Reference Table
    # -----------------------------------------------------------------------
    console.print(Panel("[bold]1. Isolation Level Reference[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]The SQL standard defines four levels ranked by strictness.\n"
        "  Higher isolation = fewer anomalies, but more lock contention and\n"
        "  lower throughput.  Choose the minimum level that keeps your data\n"
        "  correct — not the maximum.[/dim]\n"
    )

    t = Table(
        "Level", "Dirty Read", "Non-Repeatable Read", "Phantom Read", "SQLite mechanism",
        box=box.SIMPLE_HEAD,
    )
    for row in ISOLATION_MATRIX:
        t.add_row(
            row["level"],
            _bool_cell(row["dirty_read"],         "Possible", "Prevented"),
            _bool_cell(row["nonrepeatable_read"],  "Possible", "Prevented"),
            _bool_cell(row["phantom_read"],        "Possible", "Prevented"),
            f"[dim]{row['sqlite_mechanism']}[/dim]",
        )
    console.print(t)

    # -----------------------------------------------------------------------
    # Section 2: Dirty Read — always prevented in SQLite
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]2. Dirty Read — READ COMMITTED and Above[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]Dirty read: T2 reads data written by T1 before T1 commits.\n"
        "  If T1 rolls back, T2 has used data that never existed.\n\n"
        "  SQLite WAL mode makes this physically impossible: reader connections\n"
        "  access a stable snapshot and never see in-flight pages from other\n"
        "  connections.  Even the most permissive mode (autocommit) cannot\n"
        "  produce a dirty read in SQLite.[/dim]\n"
    )

    dr = demo.demonstrate_dirty_read()

    t = Table("Event", "Observation", box=box.SIMPLE_HEAD)
    t.add_row("T1 wrote uncommitted row",   str(dr["t1_wrote_uncommitted"]))
    t.add_row("T2 saw the uncommitted row", _label(dr["t2_saw_uncommitted_row"]))
    t.add_row("Dirty read occurred",        _check(not dr["anomaly_occurred"]))
    console.print(t)
    console.print(f"  [dim]Prevention: {dr['prevention']}[/dim]")

    # -----------------------------------------------------------------------
    # Section 3: Non-Repeatable Read — READ COMMITTED (autocommit)
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel(
        "[bold]3. Non-Repeatable Read — READ COMMITTED vs REPEATABLE READ[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]Non-repeatable read: T1 reads a row, T2 updates+commits it,\n"
        "  T1 reads the same row again and sees a different value.\n\n"
        "  With autocommit (READ COMMITTED), each SELECT is its own transaction.\n"
        "  Between T1's two reads, T2 commits an update → T1's second read sees\n"
        "  new data.  The read was not 'repeatable'.[/dim]\n"
    )

    nrr = demo.demonstrate_nonrepeatable_read()

    t = Table("Read", "Balance", "Note", box=box.SIMPLE_HEAD)
    t.add_row(
        "T1 first read  (before T2 commit)",
        f"{nrr['t1_first_balance']:.1f}",
        "[dim]original[/dim]",
    )
    t.add_row(
        "T1 second read (after  T2 commit)",
        f"{nrr['t1_second_balance']:.1f}",
        "[dim]T2 wrote 250.0[/dim]",
    )
    t.add_row(
        "Non-repeatable read occurred",
        "",
        _label(nrr["anomaly_occurred"]),
    )
    console.print(t)

    console.print()
    console.print(
        "  [dim]Prevention — wrap both reads in an explicit BEGIN/COMMIT.\n"
        "  SQLite WAL mode gives the transaction a stable snapshot:[/dim]\n"
    )

    rr = demo.demonstrate_repeatable_read_prevention()

    t = Table("Read", "Balance", "Note", box=box.SIMPLE_HEAD)
    t.add_row("T1 first read  (snapshot start)",   f"{rr['t1_first_balance']:.1f}",  "[dim]snapshot taken[/dim]")
    t.add_row("T2 commits update to 250.0",        "—",                              "[dim]committed while T1 reads[/dim]")
    t.add_row("T1 second read (same snapshot)",    f"{rr['t1_second_balance']:.1f}", "[dim]snapshot unchanged[/dim]")
    t.add_row("Values consistent",                 "",                               _check(rr["consistent"]))
    console.print(t)

    # -----------------------------------------------------------------------
    # Section 4: Phantom Read — READ COMMITTED vs snapshot
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel(
        "[bold]4. Phantom Read — Aggregate Queries Across INSERT[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]Phantom read: T1 runs an aggregate (COUNT/SUM/MAX) twice.\n"
        "  Between the two executions T2 inserts a new row matching the\n"
        "  predicate — the row 'appears from nowhere' (phantom).\n\n"
        "  With autocommit each COUNT is a separate transaction.  T2's\n"
        "  committed INSERT is visible to T1's second COUNT.[/dim]\n"
    )

    pr = demo.demonstrate_phantom_read()

    t = Table("Query", "Count", "Note", box=box.SIMPLE_HEAD)
    t.add_row(
        "T1 first  COUNT(genre='sci-fi')",
        str(pr["t1_first_count"]),
        "[dim]before T2 inserts[/dim]",
    )
    t.add_row(
        "T2 inserts a sci-fi movie and commits",
        "—",
        "",
    )
    t.add_row(
        "T1 second COUNT(genre='sci-fi')",
        str(pr["t1_second_count"]),
        "[dim]sees phantom row[/dim]",
    )
    t.add_row(
        "Phantom read occurred",
        "",
        _label(pr["anomaly_occurred"]),
    )
    console.print(t)

    console.print()
    console.print(
        "  [dim]Prevention — snapshot transaction (same BEGIN/COMMIT approach):[/dim]\n"
    )

    pp = demo.demonstrate_phantom_prevention()

    t = Table("Query", "Count", "Note", box=box.SIMPLE_HEAD)
    t.add_row("T1 first  COUNT (snapshot start)",    str(pp["t1_first_count"]),  "[dim]snapshot taken[/dim]")
    t.add_row("T2 inserts phantom2_1 and commits",   "—",                        "[dim]committed outside snapshot[/dim]")
    t.add_row("T1 second COUNT (same snapshot)",     str(pp["t1_second_count"]), "[dim]phantom invisible[/dim]")
    t.add_row("Phantom prevented",                   "",                          _check(pp["consistent"]))
    console.print(t)

    # -----------------------------------------------------------------------
    # Section 5: SERIALIZABLE — BEGIN EXCLUSIVE
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]5. SERIALIZABLE — BEGIN EXCLUSIVE Blocks All Concurrency[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]BEGIN EXCLUSIVE acquires an exclusive write lock immediately.\n"
        "  No other write transaction can start until T1 commits or rolls back.\n\n"
        "  In WAL mode readers can still access the stable snapshot while T1\n"
        "  holds the lock; only concurrent writers are serialised.\n"
        "  T2 also requests an EXCLUSIVE transaction here so the blocking is\n"
        "  directly observable.  Use EXCLUSIVE only when snapshot isolation is\n"
        "  insufficient — e.g. read-modify-write cycles with write-write conflicts.\n[/dim]\n"
    )

    ex = demo.demonstrate_exclusive_lock()

    t = Table("Event", "Value", box=box.SIMPLE_HEAD)
    t.add_row("T1 acquired EXCLUSIVE lock",         str(ex["t1_lock_acquired"]))
    t.add_row("T2 wait for its EXCLUSIVE lock",     f"{ex['t2_blocked_ms']:.1f} ms")
    t.add_row("T2 was blocked (> 50 ms)",           _check(ex["t2_was_blocked"]))
    t.add_row("T2 balance read after T1 commits",   f"{ex['t2_final_balance']:.1f}")
    console.print(t)
    console.print(
        "  [dim]T2's BEGIN EXCLUSIVE stalled until T1 committed — then T2 read\n"
        "  the updated balance (800.0), confirming serialised access.[/dim]"
    )

    # -----------------------------------------------------------------------
    # Section 6: Payment Processing — Lost Update
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]6. Real-World Scenario — Payment Lost Update[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]Two ATM threads simultaneously withdraw 300 from a 500-balance account.\n\n"
        "  Without serialisation (READ COMMITTED / autocommit):\n"
        "    T1 reads 500 → plans to write 200\n"
        "    T2 reads 500 → plans to write 200\n"
        "    T1 commits 200; T2 commits 200 (overwrites T1)\n"
        "    Final balance: 200  —  one withdrawal is silently LOST\n\n"
        "  With BEGIN EXCLUSIVE (SERIALIZABLE):\n"
        "    T1 gets lock, reads 500, writes 200, commits\n"
        "    T2 gets lock, reads 200, balance < 300 → REJECTED\n"
        "    Final balance: 200 with exactly one successful withdrawal[/dim]\n"
    )

    pay = demo.simulate_payment_scenario(initial_balance=500.0, withdrawal=300.0)
    unser = pay["unserialized"]
    ser   = pay["serialized"]

    console.print("[bold]Without serialisation:[/bold]")
    t = Table("", "T1", "T2", box=box.SIMPLE_HEAD)
    t.add_row("Read balance",  f"{unser['t1_read']:.1f}",  f"{unser['t2_read']:.1f}")
    t.add_row("Wrote balance", f"{unser['t1_wrote']:.1f}", f"{unser['t2_wrote']:.1f}")
    t.add_row("Final balance", f"{unser['final_balance']:.1f}", "")
    t.add_row("Withdrawals applied", str(unser["withdrawals_applied"]), "[dim](expected 1 × 300 = deduction of 300)[/dim]")
    console.print(t)
    console.print(
        f"  [red]Both transactions read 500, both computed 200, final is 200 —\n"
        f"  only 1 of 2 withdrawals is reflected (lost update).[/red]\n"
    )

    console.print("[bold]With BEGIN EXCLUSIVE (SERIALIZABLE):[/bold]")
    t = Table("Outcome", "Count", box=box.SIMPLE_HEAD)
    t.add_row("Withdrawals succeeded",              str(ser["withdrawals_succeeded"]))
    t.add_row("Withdrawals rejected (insufficient)", str(ser["withdrawals_rejected"]))
    t.add_row("Final balance",                       f"{ser['final_balance']:.1f}")
    console.print(t)
    console.print(
        f"  [green]Only one withdrawal succeeded; the second correctly read the updated\n"
        f"  balance and was rejected — no lost update, no overdraft.[/green]\n"
    )

    # -----------------------------------------------------------------------
    # Section 7: Key Takeaways
    # -----------------------------------------------------------------------
    console.print()
    console.print("[bold green]Key Transaction Isolation Takeaways:[/bold green]")
    console.print(f"  • [cyan]{LEVEL_READ_UNCOMMITTED}[/cyan]  — never use; dirty reads produce logically impossible state")
    console.print(f"  • [cyan]{LEVEL_READ_COMMITTED}[/cyan]    — PostgreSQL/Oracle default; prevents dirty reads but allows NRR + phantoms")
    console.print(f"  • [cyan]{LEVEL_REPEATABLE_READ}[/cyan]   — MySQL default; snapshot isolation prevents NRR; SQLite also prevents phantoms")
    console.print(f"  • [cyan]{LEVEL_SERIALIZABLE}[/cyan]      — strongest; BEGIN EXCLUSIVE in SQLite; serializable snapshot in PostgreSQL")
    console.print("  • [cyan]WAL snapshot isolation[/cyan]   — SQLite's WAL mode gives REPEATABLE READ semantics for free within a BEGIN block")
    console.print("  • [cyan]Lost update pattern[/cyan]      — always use EXCLUSIVE or optimistic locking for read-modify-write cycles")
    console.print("  • [cyan]Choose minimum needed[/cyan]    — SERIALIZABLE on every query kills throughput; profile before upgrading level")
    console.print(
        "  [dim]Production: PostgreSQL BEGIN ISOLATION LEVEL SERIALIZABLE; MySQL\n"
        "  SET TRANSACTION ISOLATION LEVEL; Cassandra lightweight transactions\n"
        "  (LWT) for SERIALIZABLE-equivalent compare-and-swap semantics.[/dim]"
    )

    demo.close()


if __name__ == "__main__":
    main()
