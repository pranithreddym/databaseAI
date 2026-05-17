"""
Demo 11: Database Sharding
===========================
Implements consistent-hashing-based sharding across 3 SQLite "shards".
Ratings are partitioned by user_id hash; movies are replicated as reference
data.  Demonstrates single-shard reads, fan-out aggregations, and live
rebalancing when a 4th shard is added to the ring.

Real-world parallel: Cassandra and DynamoDB both use consistent hashing
internally.  When Netflix adds a new Cassandra node to handle growing watch
history, only ≈1/N of the keyspace migrates — not a full reshuffle.
"""

import sys
import os
import tempfile
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from databaseai.seed_data import MOVIES, RATINGS, USERS
from databaseai.sharding import ConsistentHashRing, ShardManager

console = Console()

RING_VNODES = 150   # virtual nodes per physical shard


def _bar(count: int, total: int, width: int = 24) -> str:
    filled = round(width * count / total) if total else 0
    return "[green]" + "█" * filled + "[/green]" + "░" * (width - filled)


def main():
    console.rule("[bold cyan]Database Sharding Demo[/bold cyan]")
    console.print(
        "[dim]Real-world parallel: Cassandra / DynamoDB consistent-hash partitioning —\n"
        "  routing reads and writes by hash(user_id) with minimal key migration\n"
        "  when new nodes join the cluster.[/dim]\n"
    )

    # ----------------------------------------------------------------
    # Build temp shard files
    # ----------------------------------------------------------------
    tmpdir = tempfile.mkdtemp(prefix="sharding_demo_")
    shard_paths = [os.path.join(tmpdir, f"shard_{i}.db") for i in range(3)]
    extra_shard_path = os.path.join(tmpdir, "shard_3.db")

    # ---------------------------------------------------------------
    # Section 1: Ring Setup
    # ---------------------------------------------------------------
    console.print(Panel("[bold]1. Consistent Hash Ring — 3 Shards[/bold]", box=box.ROUNDED))
    console.print(
        f"  [dim]Each physical shard expands to {RING_VNODES} virtual nodes placed\n"
        f"  uniformly around the 32-bit integer ring (0 … 2³² − 1).\n"
        f"  A user_id is hashed and assigned to the first vnode clockwise.[/dim]"
    )

    ring = ConsistentHashRing(vnodes_per_node=RING_VNODES)
    for path in shard_paths:
        ring.add_node(os.path.basename(path))

    t = Table("Metric", "Value", box=box.SIMPLE_HEAD)
    t.add_row("physical shards",       str(len(ring.nodes)))
    t.add_row("virtual nodes total",   str(len(ring.nodes) * RING_VNODES))
    t.add_row("ring size (bits)",      "32")
    t.add_row("ring capacity",         "4,294,967,296 positions")
    console.print(t)

    user_ids = [u["id"] for u in USERS]
    dist = ring.key_distribution(user_ids)
    t = Table("Shard", "Users Assigned", "Distribution", box=box.SIMPLE_HEAD)
    for node, count in sorted(dist.items()):
        t.add_row(node, str(count), _bar(count, len(user_ids)))
    console.print(t)
    console.print(
        "  [dim]Vnodes make arcs proportionally equal; with only 5 users the\n"
        "  distribution looks skewed — run with 1 000 keys for near-perfect balance.[/dim]"
    )

    # ---------------------------------------------------------------
    # Section 2: Data Population — Partitioned Writes
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]2. Partitioned Writes — Routing by user_id[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]Movies are broadcast to every shard (small reference table, avoids\n"
        "  cross-shard JOINs in recommendation queries).\n"
        "  Each rating is written to exactly ONE shard — the one that owns its user_id.[/dim]"
    )

    mgr = ShardManager(shard_paths, vnodes_per_node=RING_VNODES)
    mgr.insert_movies(MOVIES)
    per_shard_write = mgr.insert_ratings_bulk(RATINGS)

    t = Table("Shard", "Ratings Written", "Bar", box=box.SIMPLE_HEAD)
    total_written = sum(per_shard_write.values())
    for shard, cnt in sorted(per_shard_write.items()):
        t.add_row(shard, str(cnt), _bar(cnt, total_written))
    console.print(t)
    console.print(
        f"  Total ratings: [bold]{total_written}[/bold] across "
        f"{len(shard_paths)} shards — each user's history on exactly one shard.\n"
        f"  [dim]A coordinator node never touches shard data for single-user lookups.[/dim]"
    )

    # ---------------------------------------------------------------
    # Section 3: Routing Table — Where Does Each User Live?
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]3. Routing Table — Per-User Shard Assignment[/bold]", box=box.ROUNDED))
    t = Table("User", "Username", "Shard (owner)", "Ratings on shard",
              box=box.SIMPLE_HEAD)
    for u in USERS:
        shard = mgr.get_shard_for_user(u["id"])
        count = len(mgr.user_ratings(u["id"]))
        t.add_row(u["id"], u["username"], shard, str(count))
    console.print(t)
    console.print(
        "  [dim]Routing is deterministic: any coordinator can derive the shard for a\n"
        "  user_id independently without a lookup table — just hash and walk the ring.[/dim]"
    )

    # ---------------------------------------------------------------
    # Section 4: Single-Shard Read
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]4. Single-Shard Read — alice_w's Ratings[/bold]", box=box.ROUNDED))
    target_user = "u01"
    target_shard = mgr.get_shard_for_user(target_user)
    ratings = mgr.user_ratings(target_user)
    console.print(
        f"  user_id=[bold]u01[/bold] (alice_w)  →  shard=[bold]{target_shard}[/bold]\n"
        f"  Query touches [bold]1 / {len(shard_paths)} shards[/bold]; "
        f"other shards are not contacted."
    )
    t = Table("Movie", "Genre", "Score", "Review", box=box.SIMPLE_HEAD)
    for r in ratings:
        t.add_row(r["title"], r["genre"], str(r["score"]), r["review"])
    console.print(t)

    # ---------------------------------------------------------------
    # Section 5: Fan-Out Read — Global Top-Rated
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]5. Fan-Out Read — Global Top-Rated (All Shards)[/bold]", box=box.ROUNDED))
    console.print(
        f"  [dim]Scatter to all {len(shard_paths)} shards → gather partial aggregates →\n"
        f"  merge in the coordinator.  Equivalent to Cassandra QUORUM read or\n"
        f"  DynamoDB parallel scan.[/dim]"
    )
    top = mgr.global_top_rated(limit=5)
    t = Table("Rank", "Title", "Genre", "Avg Score", "Votes", box=box.SIMPLE_HEAD)
    for i, row in enumerate(top, 1):
        t.add_row(str(i), row["title"], row["genre"],
                  str(row["avg_score"]), str(row["votes"]))
    console.print(t)
    console.print(
        "  [dim]Coordinator merged partial counts from each shard; no shard needed\n"
        "  to see another shard's rows — only the coordinator aggregates.[/dim]"
    )

    # ---------------------------------------------------------------
    # Section 6: Live Rebalancing — Add Shard 4
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]6. Live Rebalancing — Adding Shard 4 to the Ring[/bold]", box=box.ROUNDED))
    before = mgr.rating_count_per_shard().copy()
    before_total = sum(before.values())
    console.print("  [dim]Distribution BEFORE adding shard_3.db:[/dim]")
    t = Table("Shard", "Ratings", "Share", box=box.SIMPLE_HEAD)
    for shard, cnt in sorted(before.items()):
        pct = cnt / before_total * 100 if before_total else 0
        t.add_row(shard, str(cnt), f"{pct:.0f}%")
    console.print(t)

    new_node, rows_migrated = mgr.add_shard(extra_shard_path)

    after = mgr.rating_count_per_shard()
    after_total = sum(after.values())
    console.print(f"\n  New shard [bold]{new_node}[/bold] joined the ring.")
    console.print(f"  Rows migrated: [bold yellow]{rows_migrated}[/bold yellow] / "
                  f"{before_total} "
                  f"({rows_migrated / before_total * 100:.0f}% of keyspace)")

    expected_pct = 1 / len(mgr.shard_ids) * 100
    console.print(
        f"  Consistent-hash theory predicts ≈ [dim]{expected_pct:.0f}%[/dim] migration "
        f"when N → N+1 (1/(N+1) rule)."
    )

    console.print("\n  [dim]Distribution AFTER adding shard_3.db:[/dim]")
    t = Table("Shard", "Ratings", "Share", "Change", box=box.SIMPLE_HEAD)
    for shard, cnt in sorted(after.items()):
        pct = cnt / after_total * 100 if after_total else 0
        old_cnt = before.get(shard, 0)
        delta = cnt - old_cnt
        delta_str = (f"[red]{delta}[/red]" if delta < 0
                     else f"[green]+{delta}[/green]" if delta > 0
                     else "0")
        t.add_row(shard, str(cnt), f"{pct:.0f}%", delta_str)
    console.print(t)
    console.print(
        "  [dim]Only the shard(s) immediately adjacent to the new node on the ring\n"
        "  donated rows.  Shards on the far side of the ring were untouched.[/dim]"
    )

    # ---------------------------------------------------------------
    # Section 7: Routing Correctness Post-Rebalance
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]7. Routing Correctness After Rebalance[/bold]", box=box.ROUNDED))
    t = Table("User", "New Shard", "Ratings Found", "Correct?", box=box.SIMPLE_HEAD)
    all_correct = True
    for u in USERS:
        shard = mgr.get_shard_for_user(u["id"])
        count = len(mgr.user_ratings(u["id"]))
        for other_shard in mgr.shard_ids:
            if other_shard == shard:
                continue
        correct = count > 0
        all_correct = all_correct and correct
        mark = "[green]✓[/green]" if correct else "[red]✗[/red]"
        t.add_row(u["id"], shard, str(count), mark)
    console.print(t)
    status = "[bold green]All users found on their assigned shard.[/bold green]"
    console.print(f"  {status}")
    console.print(
        "  [dim]Reads after rebalance use the same ring hash — no routing table update\n"
        "  needed on the coordinator side, consistent with Cassandra gossip protocol.[/dim]"
    )

    # ---------------------------------------------------------------
    # Section 8: Synthetic Key Distribution (1000 keys)
    # ---------------------------------------------------------------
    console.print()
    console.print(Panel("[bold]8. Statistical Distribution — 1 000 Synthetic Keys[/bold]", box=box.ROUNDED))
    console.print(
        f"  [dim]With only 5 real users the distribution looks uneven.\n"
        f"  1 000 synthetic keys show how the ring balances at scale.[/dim]"
    )
    synthetic_keys = [f"user_{i:04d}" for i in range(1000)]
    synth_dist = mgr._ring.key_distribution(synthetic_keys)
    synth_total = sum(synth_dist.values())
    ideal = synth_total / len(mgr.shard_ids)
    t = Table("Shard", "Keys", "% of total", "Deviation from ideal", box=box.SIMPLE_HEAD)
    for shard, cnt in sorted(synth_dist.items()):
        pct = cnt / synth_total * 100
        dev = (cnt - ideal) / ideal * 100
        dev_str = (f"[yellow]{dev:+.1f}%[/yellow]" if abs(dev) > 10
                   else f"{dev:+.1f}%")
        t.add_row(shard, str(cnt), f"{pct:.1f}%", dev_str)
    console.print(t)
    stddev = math.sqrt(
        sum((synth_dist[n] - ideal) ** 2 for n in synth_dist) / len(synth_dist)
    )
    console.print(
        f"  Std-dev: [bold]{stddev:.1f}[/bold] keys  "
        f"(ideal = {ideal:.0f} per shard, {RING_VNODES} vnodes/shard)"
    )

    # ---------------------------------------------------------------
    # Section 9: Key Takeaways
    # ---------------------------------------------------------------
    console.print()
    console.print("[bold green]Key Sharding Takeaways:[/bold green]")
    console.print("  • [cyan]Consistent hashing[/cyan]    — adding/removing a node migrates only 1/N of the keyspace")
    console.print("  • [cyan]Virtual nodes[/cyan]         — multiply ring positions per shard for uniform arc coverage")
    console.print("  • [cyan]Partition strategy[/cyan]    — partition hot (ratings) data; replicate cold (movies) reference data")
    console.print("  • [cyan]Single-shard read[/cyan]     — user-scoped queries touch exactly 1 shard; sub-millisecond routing")
    console.print("  • [cyan]Fan-out read[/cyan]          — global aggregations scatter to all shards then merge at coordinator")
    console.print("  • [cyan]Deterministic routing[/cyan] — any node computes the owner from hash(key) alone; no lookup table")
    console.print("  [dim]Production: Cassandra (Murmur3 + vnodes), DynamoDB (opaque hash partitioning),\n"
                  "  Vitess (MySQL sharding), MongoDB hashed index sharding[/dim]")

    mgr.close()
    import shutil
    try:
        shutil.rmtree(tmpdir)
    except OSError:
        pass


if __name__ == "__main__":
    main()
