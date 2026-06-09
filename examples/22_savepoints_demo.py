"""
Demo 22: Savepoints & Nested Rollbacks
=======================================
Demo 13 demonstrated isolation levels — how concurrent transactions see each
other's work.  This demo explores the *intra-transaction* control surface:
SQLite SAVEPOINT / RELEASE / ROLLBACK TO, which let you undo individual steps
inside a single transaction without aborting the whole unit of work.

Real-world parallel: a streaming service's subscription purchase pipeline
(create order -> reserve slots -> charge payment -> activate subscription) where
a transient payment-gateway timeout in step 3 should retry that step alone —
not roll back the audit records already written in steps 1 and 2.  The same
per-row-savepoint pattern drives fault-tolerant batch ETL loaders that skip
malformed records while committing the rest of the batch.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from databaseai.seed_data import MOVIES
from databaseai.savepoints import SavepointDemo

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _outcome_cell(outcome: str) -> str:
    if outcome in ("success", "success_after_retry"):
        return f"[green]{outcome}[/green]"
    if outcome == "retrying":
        return f"[yellow]{outcome}[/yellow]"
    return f"[red]{outcome}[/red]"


def _bool_cell(flag: bool, yes: str = "YES", no: str = "NO") -> str:
    return f"[green]{yes}[/green]" if flag else f"[red]{no}[/red]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    console.rule("[bold cyan]Savepoints & Nested Rollbacks[/bold cyan]")
    console.print(
        "[dim]Real-world parallel: multi-step purchase flows, nested atomic blocks\n"
        "  (Django transaction.atomic), and fault-tolerant batch ETL loaders that\n"
        "  skip bad rows without aborting the whole batch.[/dim]\n"
    )

    demo = SavepointDemo()
    demo.seed(MOVIES)

    # -----------------------------------------------------------------------
    # Section 1: The SAVEPOINT / ROLLBACK TO / RELEASE triad
    # -----------------------------------------------------------------------
    console.print(Panel(
        "[bold]1. SAVEPOINT Basics — Partial Rollback Within a Transaction[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]A SAVEPOINT names a checkpoint inside an open transaction.\n"
        "  ROLLBACK TO sp rewinds the database to that checkpoint — removing\n"
        "  all rows inserted or updated after sp — while keeping the outer\n"
        "  transaction alive.  RELEASE sp then merges the (now-clean)\n"
        "  savepoint scope into the enclosing transaction.\n\n"
        "  Scenario:\n"
        "    BEGIN\n"
        "    INSERT m01              <- persisted in outer txn\n"
        "    SAVEPOINT sp_after_a\n"
        "    INSERT m02              <- inside savepoint\n"
        "    ROLLBACK TO sp_after_a  <- m02 undone; txn still alive\n"
        "    RELEASE sp_after_a\n"
        "    COMMIT                  <- only m01 reaches disk[/dim]\n"
    )

    sp = demo.basic_savepoint(user_id="u01")

    t = Table("State", "Row count in watchlist", box=box.SIMPLE_HEAD)
    t.add_row("After INSERT m01 (before savepoint)", str(sp["count_after_insert_a"]))
    t.add_row("After INSERT m02 (inside savepoint)", str(sp["count_after_insert_b"]))
    t.add_row("After ROLLBACK TO sp_after_a",        str(sp["count_after_rollback"]))
    t.add_row("After COMMIT (final committed state)", str(sp["final_count"]))
    console.print(t)

    t2 = Table("Movie", "In committed watchlist?", box=box.SIMPLE_HEAD)
    t2.add_row("m01 (inserted before savepoint)", _bool_cell(sp["item_a_committed"]))
    t2.add_row("m02 (inserted inside savepoint)", _bool_cell(sp["item_b_committed"], "YES (wrong)", "NO (correct)"))
    console.print(t2)
    console.print(
        "  [dim]ROLLBACK TO removed m02 without touching m01 — the outer BEGIN/COMMIT\n"
        "  boundary was never crossed.  A full ROLLBACK would have discarded m01 too.[/dim]"
    )

    # -----------------------------------------------------------------------
    # Section 2: Nested SAVEPOINTs
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel(
        "[bold]2. Nested SAVEPOINTs — Outer Rollback Discards All Inner Layers[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]Savepoints stack.  Every new SAVEPOINT adds a layer on top;\n"
        "  ROLLBACK TO an outer savepoint rewinds to THAT layer and implicitly\n"
        "  destroys every inner savepoint above it.\n\n"
        "  Stack diagram:\n"
        "    BEGIN\n"
        "      INSERT m01\n"
        "      SAVEPOINT outer_sp   <- checkpoint A\n"
        "        INSERT m02\n"
        "        SAVEPOINT inner_sp <- checkpoint B (inside A)\n"
        "          INSERT m03\n"
        "      ROLLBACK TO outer_sp <- rewinds to A; inner_sp + m02 + m03 gone\n"
        "      RELEASE outer_sp\n"
        "    COMMIT                 <- only m01 reaches disk[/dim]\n"
    )

    ns = demo.nested_savepoints(user_id="u02")

    t = Table("State", "Row count in watchlist", box=box.SIMPLE_HEAD)
    t.add_row("After INSERT m01",                       str(ns["count_after_m01"]))
    t.add_row("After INSERT m02 + m03 (nested)",        str(ns["count_with_all_three"]))
    t.add_row("After ROLLBACK TO outer_sp",             str(ns["count_after_outer_rollback"]))
    console.print(t)

    t2 = Table("Outcome", "Value", box=box.SIMPLE_HEAD)
    t2.add_row("Movies that survived COMMIT", ", ".join(ns["survived_movies"]) or "(none)")
    t2.add_row("Only m01 survived", _bool_cell(ns["only_m01_survived"]))
    console.print(t2)
    console.print(
        "  [dim]Rolling back to outer_sp tore down the inner_sp frame along with it —\n"
        "  SQLite does not require explicit cleanup of inner savepoints when an outer\n"
        "  one is rolled back.  This matches Python's exception-propagation model:\n"
        "  an uncaught exception in an inner atomic() block propagates outward and\n"
        "  triggers the outer block's rollback too.[/dim]"
    )

    # -----------------------------------------------------------------------
    # Section 3: Subscription purchase flow — happy path
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel(
        "[bold]3. Subscription Purchase Flow — Per-Step Savepoints, Happy Path[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]Each step in a multi-stage workflow wraps its SQL in a SAVEPOINT.\n"
        "  On success RELEASE merges the step into the outer transaction.\n"
        "  Steps: create_order -> reserve_slots -> charge_payment -> activate_sub[/dim]\n"
    )

    happy = demo.purchase_flow(order_id="ord_happy", user_id="u01", plan="premium")

    t = Table("Step", "Outcome", box=box.SIMPLE_HEAD)
    for s in happy["steps"]:
        t.add_row(s["step"], _outcome_cell(s["outcome"]))
    console.print(t)

    if happy["order_status"]:
        console.print(
            f"  Order status  : [green]{happy['order_status']['status']}[/green]\n"
            f"  Subscription  : [green]{happy['subscription']['status'] if happy['subscription'] else 'none'}[/green]"
        )
    console.print(
        f"  All steps succeeded: {_bool_cell(happy['all_steps_succeeded'])}\n"
    )

    # -----------------------------------------------------------------------
    # Section 4: Subscription purchase flow — transient failure + retry
    # -----------------------------------------------------------------------
    console.print(Panel(
        "[bold]4. Purchase Flow — Transient Payment Failure, Savepoint Retry[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]The charge_payment step fails on its first attempt (simulating a\n"
        "  gateway timeout).  ROLLBACK TO that step's savepoint rewinds only the\n"
        "  payment work; create_order and reserve_slots results are intact.\n"
        "  A second attempt of charge_payment succeeds, and the outer transaction\n"
        "  commits all four steps as if no failure occurred.\n\n"
        "  This is exactly how Stripe's charge pipeline wraps each stage: a\n"
        "  transient failure in the card-charge call triggers a ROLLBACK TO\n"
        "  without discarding the fraud-check record written earlier.[/dim]\n"
    )

    retry = demo.purchase_flow(
        order_id="ord_retry",
        user_id="u03",
        plan="standard",
        fail_at="charge_payment",
    )

    t = Table("Step", "Outcome", "Note", box=box.SIMPLE_HEAD)
    for s in retry["steps"]:
        note = s.get("error", "") if s["outcome"] == "retrying" else ""
        t.add_row(s["step"], _outcome_cell(s["outcome"]), f"[dim]{note}[/dim]")
    console.print(t)

    console.print(
        f"  Final order status  : [green]{retry['order_status']['status'] if retry['order_status'] else 'none'}[/green]\n"
        f"  Subscription active : {_bool_cell(retry['subscription'] is not None and retry['subscription']['status'] == 'active')}\n"
        f"  All steps succeeded : {_bool_cell(retry['all_steps_succeeded'])}"
    )
    console.print(
        "\n  [dim]The failed charge_payment step was retried within the same outer\n"
        "  transaction — create_order and reserve_slots were never rolled back.[/dim]"
    )

    # -----------------------------------------------------------------------
    # Section 5: Batch ingestion with per-row savepoints
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel(
        "[bold]5. Batch Content Ingestion — Per-Row Savepoints, Skip Bad Records[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]A catalog import job ingests 8 title records in a single transaction.\n"
        "  Two records are flagged as invalid (missing required fields, simulated\n"
        "  via a validation error).  Each row gets its own savepoint:\n\n"
        "    for each record:\n"
        "      SAVEPOINT sp_row_{id}\n"
        "      INSERT ...\n"
        "      if validation error:  ROLLBACK TO sp_row_{id}  <- skip row\n"
        "      else:                 RELEASE sp_row_{id}       <- keep row\n"
        "    COMMIT\n\n"
        "  The batch commits all 6 good records; the 2 bad ones are skipped\n"
        "  without aborting the batch.  This is the pattern used by Airflow\n"
        "  ETL operators and Spark JDBC writers with error tolerances.[/dim]\n"
    )

    batch_records = [
        {"movie_id": "new01", "title": "Dune: Part Two",         "genre": "sci-fi",    "year": 2024, "rating": 4.8},
        {"movie_id": "new02", "title": "Oppenheimer",            "genre": "drama",     "year": 2023, "rating": 4.7},
        {"movie_id": "new03", "title": "Poor Things",            "genre": "drama",     "year": 2023, "rating": 4.3},
        {"movie_id": "new04", "title": "Past Lives",             "genre": "drama",     "year": 2023, "rating": 4.5},
        {"movie_id": "bad01", "title": "",                        "genre": None,        "year": None, "rating": None},
        {"movie_id": "new05", "title": "The Zone of Interest",   "genre": "drama",     "year": 2023, "rating": 4.2},
        {"movie_id": "bad02", "title": "",                        "genre": None,        "year": None, "rating": None},
        {"movie_id": "new06", "title": "American Fiction",       "genre": "comedy",    "year": 2023, "rating": 4.1},
    ]
    fail_ids = {"bad01", "bad02"}

    result = demo.batch_ingest(batch_records, fail_ids=fail_ids)

    t = Table("Record ID", "Title", "Outcome", box=box.SIMPLE_HEAD)
    for rec in batch_records:
        rid = rec["movie_id"]
        title = rec["title"] or "[dim](no title — invalid)[/dim]"
        if rid in result["succeeded"]:
            outcome = "[green]committed[/green]"
        else:
            outcome = "[yellow]skipped (bad data)[/yellow]"
        t.add_row(rid, title, outcome)
    console.print(t)

    t2 = Table("Metric", "Value", box=box.SIMPLE_HEAD)
    t2.add_row("Total records",    str(result["total"]))
    t2.add_row("Committed",        f"[green]{result['success_count']}[/green]")
    t2.add_row("Skipped (errors)", f"[yellow]{result['skip_count']}[/yellow]")
    console.print(t2)
    console.print(
        "  [dim]6 of 8 records were committed in a single transaction.  The 2 bad\n"
        "  records' savepoints were rolled back individually — the outer BEGIN was\n"
        "  never affected.  A full ROLLBACK on error would have wasted all 6 good\n"
        "  records and forced a full retry of the batch.[/dim]"
    )

    # -----------------------------------------------------------------------
    # Section 6: ROLLBACK TO SAVEPOINT vs full ROLLBACK
    # -----------------------------------------------------------------------
    console.print()
    console.print(Panel(
        "[bold]6. ROLLBACK TO SAVEPOINT vs Full ROLLBACK — The Key Distinction[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]Scenario A — savepoint partial rollback:\n"
        "    BEGIN -> INSERT m04 -> SAVEPOINT sp -> INSERT m05 ->\n"
        "    ROLLBACK TO sp -> INSERT m06 -> RELEASE sp -> COMMIT\n"
        "    Persisted: {m04, m06}   (m05 was removed at the savepoint)\n\n"
        "  Scenario B — full ROLLBACK:\n"
        "    BEGIN -> INSERT m04 -> INSERT m05 -> ROLLBACK\n"
        "    Persisted: {}           (entire transaction discarded)[/dim]\n"
    )

    vs = demo.savepoint_vs_full_rollback(user_id="u04")

    t = Table("Scenario", "SQL path", "Movies persisted", "Count", box=box.SIMPLE_HEAD)
    t.add_row(
        "A — savepoint rollback",
        "BEGIN -> sp -> ROLLBACK TO sp -> COMMIT",
        ", ".join(vs["scenario_a_persisted"]) or "(none)",
        str(vs["scenario_a_count"]),
    )
    t.add_row(
        "B — full ROLLBACK",
        "BEGIN -> ROLLBACK",
        ", ".join(vs["scenario_b_persisted"]) or "(none)",
        str(vs["scenario_b_count"]),
    )
    console.print(t)
    console.print(
        "  [dim]Scenario A committed 2 movies (m04 and m06); m05 was the 'mistake'\n"
        "  that the savepoint undid.  Scenario B committed nothing — the full\n"
        "  ROLLBACK discarded every row inserted since BEGIN.[/dim]"
    )

    # -----------------------------------------------------------------------
    # Section 7: Key Takeaways
    # -----------------------------------------------------------------------
    console.print()
    console.print("[bold green]Key Savepoint Takeaways:[/bold green]")
    console.print("  • [cyan]SAVEPOINT[/cyan]          — creates a named intra-transaction checkpoint")
    console.print("  • [cyan]ROLLBACK TO[/cyan]        — partial undo to that checkpoint; outer transaction stays alive")
    console.print("  • [cyan]RELEASE[/cyan]            — merges savepoint's work into the enclosing transaction; removes the checkpoint")
    console.print("  • [cyan]Nesting[/cyan]            — rolling back an outer savepoint destroys all inner ones too")
    console.print("  • [cyan]Per-step retry[/cyan]     — transient failure in step N rolls back only step N; steps 1..N-1 are unaffected")
    console.print("  • [cyan]Batch skip[/cyan]         — per-row savepoints let a bulk loader commit good rows and skip bad ones")
    console.print("  • [cyan]vs full ROLLBACK[/cyan]   — ROLLBACK TO keeps the transaction alive; full ROLLBACK discards everything")
    console.print(
        "  [dim]Production: Django's transaction.atomic() uses savepoints for every\n"
        "  nested block; PostgreSQL PL/pgSQL EXCEPTION clauses rely on implicit\n"
        "  savepoints; Stripe's pipeline savepoints every charge stage so a\n"
        "  gateway timeout triggers a per-step retry, not a full order cancellation.[/dim]"
    )

    demo.close()


if __name__ == "__main__":
    main()
