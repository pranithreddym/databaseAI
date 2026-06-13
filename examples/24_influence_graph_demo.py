"""
Demo 24: Graph Database — Directed Influence Graph & PageRank
=============================================================
Demo 08 modelled movies as an undirected adjacency list; Demo 16 built a
bipartite user-movie collaborative filter.  This demo takes the third angle:
a *directed weighted graph* where an edge A→B captures the empirical influence
one movie exerts on what co-viewers watch next.  Three graph-ranking algorithms
are compared — PageRank, Personalised PageRank (RWR), and HITS — each surfacing
a different facet of movie authority in the recommendation network.

Real-world parallel: YouTube's watch-next graph.  Every time a viewer clicks
"Up Next" or autoplay kicks in, a directed edge is strengthened between the
two videos.  A PageRank variant over this graph determines which videos surface
in the algorithmic feed; Pinterest's Pixie algorithm applies the same Random
Walk with Restart idea to pin-board graphs at billion-node scale.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from databaseai.seed_data import MOVIES, RATINGS
from databaseai.influence_graph import InfluenceGraph

console = Console()

_TITLE = {m["id"]: m["title"] for m in MOVIES}
_GENRE = {m["id"]: m["genre"] for m in MOVIES}


def _truncate(s: str, n: int = 30) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _bar(score: float, max_score: float, width: int = 16) -> str:
    filled = int(round(score / max_score * width)) if max_score > 0 else 0
    return "[cyan]" + "█" * filled + "░" * (width - filled) + "[/cyan]"


def _build_graph() -> InfluenceGraph:
    g = InfluenceGraph()
    g.build_from_seed(MOVIES, RATINGS)
    return g


def main():
    console.rule("[bold cyan]Graph Database Demo 24 — Directed Influence Graph & PageRank[/bold cyan]")
    console.print(
        "[dim]Real-world parallel: YouTube watch-next graph — PageRank variants over "
        "directed co-view edges determine which videos appear in the algorithmic feed; "
        "Pinterest Pixie uses the same Random Walk with Restart at billion-node scale.[/dim]\n"
    )

    g = _build_graph()

    # ── Section 1: Graph Construction ────────────────────────────────────────────
    console.print(Panel("[bold]1. Graph Construction — Directed Co-Rating Influence Graph[/bold]", box=box.ROUNDED))
    console.print(
        f"  [green]✓[/green] {g.node_count()} movie nodes  "
        f"| {g.edge_count()} directed edges\n"
        "  [dim]Schema: nodes(node_id, label, genre) + edges(from_node, to_node, weight)[/dim]\n"
        "  [dim]Edge A→B weight = Σ rating(B) over all users who rated both A and B.[/dim]\n"
        "  [dim]Two indexes (idx_inf_from, idx_inf_to) make both outgoing and incoming "
        "neighbor lookups O(log E + k) — critical for iterative PageRank updates.[/dim]"
    )

    # ── Section 2: Degree Analysis ───────────────────────────────────────────
    console.print()
    console.print(Panel("[bold]2. Degree Centrality — Gateways vs. Destinations[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]In-degree = number of movies that lead HERE (a destination).\n"
        "  Out-degree = number of movies this film leads TO (a gateway).\n"
        "  High in-degree → many paths converge; high out-degree → pivotal gateway film.[/dim]"
    )
    t = Table("Movie", "Genre", "In-degree", "Out-degree", "Total", box=box.SIMPLE_HEAD)
    for row in g.degree_centrality()[:10]:
        t.add_row(
            _truncate(row["label"]),
            row["genre"] or "",
            str(row["in_degree"]),
            str(row["out_degree"]),
            str(row["total_degree"]),
        )
    console.print(t)

    # ── Section 3: Global PageRank ───────────────────────────────────────────
    console.print()
    console.print(Panel("[bold]3. Global PageRank — Weighted Authority Across the Graph[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]PageRank flows through directed edges proportional to edge weight, "
        "so a highly-rated destination movie accumulates more score than a merely "
        "popular one.  Damping factor 0.85 means 15 % of rank is redistributed "
        "uniformly (the teleport that prevents rank sinks and ensures convergence).[/dim]"
    )
    pr = g.pagerank(damping=0.85)
    top_pr = g.top_n_by_score(pr, n=10)
    max_pr = max(pr.values()) if pr else 1.0
    t = Table("Rank", "Movie", "Genre", "PageRank", "Bar", box=box.SIMPLE_HEAD)
    for i, row in enumerate(top_pr, 1):
        t.add_row(
            str(i),
            _truncate(row["label"]),
            row["genre"] or "",
            f"{row['score']:.5f}",
            _bar(row["score"], max_pr),
        )
    console.print(t)

    # ── Section 4: PageRank vs Degree ──────────────────────────────────────────
    console.print()
    console.print(Panel("[bold]4. PageRank vs. In-Degree — When They Diverge[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]In-degree counts raw links; PageRank weights them by the authority "
        "of the linking node.  A movie linked from a high-PR hub scores higher than "
        "one linked from many low-PR films — the same intuition behind Google's "
        "original citation-weighting insight.[/dim]"
    )
    degree_rows = g.degree_centrality()
    pr_rank = {nd: i + 1 for i, (nd, _) in enumerate(sorted(pr.items(), key=lambda x: -x[1]))}
    in_deg_sorted = sorted(degree_rows, key=lambda x: -x["in_degree"])[:8]
    t = Table("Movie", "In-degree rank", "PageRank rank", "In-degree", "PR score", box=box.SIMPLE_HEAD)
    for i, row in enumerate(in_deg_sorted, 1):
        nid = row["node_id"]
        prr = pr_rank.get(nid, "—")
        diff = i - prr if isinstance(prr, int) else 0
        diff_str = (
            f"[green]+{diff}[/green]" if diff > 0
            else (f"[red]{diff}[/red]" if diff < 0 else "[dim]0[/dim]")
        )
        t.add_row(
            _truncate(row["label"]),
            str(i),
            f"{prr} ({diff_str})",
            str(row["in_degree"]),
            f"{pr.get(nid, 0):.5f}",
        )
    console.print(t)

    # ── Section 5: Personalized PageRank ─────────────────────────────────────────
    console.print()
    console.print(Panel("[bold]5. Personalized PageRank — Seed-Based Recommendation[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]Personalized PageRank concentrates all teleport mass on a single seed "
        "movie.  The resulting scores measure proximity to the seed in the influence "
        "graph — exactly how Pinterest Pixie generates per-pin recommendations "
        "without computing a full n×n similarity matrix.[/dim]"
    )
    seeds = [("m01", "Inception"), ("m07", "The Shawshank Redemption"), ("m05", "Parasite")]
    for seed_id, seed_title in seeds:
        ppr = g.pagerank(damping=0.85, personalized={seed_id: 1.0})
        top_ppr = [r for r in g.top_n_by_score(ppr, n=6) if r["node_id"] != seed_id][:5]
        console.print(f"\n  [bold cyan]Seed: {seed_title}[/bold cyan]")
        t = Table("Rank", "Recommended Movie", "Genre", "PPR Score", box=box.SIMPLE_HEAD)
        for i, row in enumerate(top_ppr, 1):
            t.add_row(str(i), _truncate(row["label"]), row["genre"] or "", f"{row['score']:.5f}")
        console.print(t)

    # ── Section 6: Random Walk with Restart ──────────────────────────────────────
    console.print()
    console.print(Panel("[bold]6. Random Walk with Restart — Monte Carlo Approximation[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]RWR simulates 5 000 steps of a random walker that resets to the seed "
        "with probability 0.15.  Visit-count fractions approximate Personalized "
        "PageRank.  The stochastic nature makes it sub-linear: you only visit a "
        "small neighbourhood instead of touching all nodes — Pinterest Pixie exploits "
        "this to run in microseconds on a billion-node graph.[/dim]"
    )
    rwr_seed = "m01"  # Inception
    rwr_scores = g.random_walk_with_restart(rwr_seed, steps=5000, restart_prob=0.15, rng_seed=42)
    top_rwr = [r for r in g.top_n_by_score(rwr_scores, n=6) if r["node_id"] != rwr_seed][:5]
    ppr_scores = g.pagerank(damping=0.85, personalized={rwr_seed: 1.0})
    ppr_rank_map = {nd: i + 1 for i, (nd, _) in enumerate(
        sorted(ppr_scores.items(), key=lambda x: -x[1])
    )}
    console.print(f"\n  [bold cyan]Seed: Inception (m01)  — RWR vs. Personalized PageRank[/bold cyan]")
    t = Table("Rank (RWR)", "Movie", "Genre", "RWR Visits %", "PPR Rank", box=box.SIMPLE_HEAD)
    for i, row in enumerate(top_rwr, 1):
        nid = row["node_id"]
        pprank = ppr_rank_map.get(nid, "—")
        t.add_row(
            str(i),
            _truncate(row["label"]),
            row["genre"] or "",
            f"{row['score'] * 100:.2f}%",
            str(pprank),
        )
    console.print(t)
    console.print(
        "  [dim]RWR and PPR rankings generally agree; small differences are Monte Carlo "
        "sampling noise that shrinks as step count increases.[/dim]"
    )

    # ── Section 7: HITS ───────────────────────────────────────────────────────
    console.print()
    console.print(Panel("[bold]7. HITS — Hub Score vs. Authority Score[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]HITS separates two roles:\n"
        "    Hub       → a film that reliably leads viewers to acclaimed movies.\n"
        "    Authority → a film that many hub films point to; the 'destination'.\n"
        "  A movie can score high on both (a gateway AND a destination) or excel at "
        "only one.  The divergence reveals viewing-pattern asymmetries invisible to "
        "undirected degree centrality.[/dim]"
    )
    hubs, auths = g.hits()
    top_auth = g.top_n_by_score(auths, n=5)
    top_hub = g.top_n_by_score(hubs, n=5)
    t = Table(
        "Auth Rank", "Authority Movie", "Auth Score",
        "Hub Rank",  "Hub Movie",       "Hub Score",
        box=box.SIMPLE_HEAD,
    )
    for i in range(5):
        a = top_auth[i] if i < len(top_auth) else {}
        h = top_hub[i]  if i < len(top_hub)  else {}
        t.add_row(
            str(i + 1),
            _truncate(a.get("label", "—"), 22),
            f"{a.get('score', 0):.5f}",
            str(i + 1),
            _truncate(h.get("label", "—"), 22),
            f"{h.get('score', 0):.5f}",
        )
    console.print(t)

    # ── Section 8: PageRank Convergence ─────────────────────────────────────────
    console.print()
    console.print(Panel("[bold]8. PageRank Convergence — Score Stabilisation Over Iterations[/bold]", box=box.ROUNDED))
    console.print(
        "  [dim]Power iteration starts every node at 1/N and repeatedly redistributes "
        "rank along weighted edges.  The scores converge to the dominant eigenvector of "
        "the transition matrix, usually within 20–40 iterations for small graphs.  "
        "Google's original PageRank ran on a 25-million-page web graph and converged "
        "in roughly 52 iterations.[/dim]"
    )
    top3 = [row["node_id"] for row in top_pr[:3]]
    history = g.pagerank_history(damping=0.85, max_iter=30, track_nodes=top3)
    t = Table("Iteration", *[_truncate(_TITLE[nid], 18) for nid in top3], box=box.SIMPLE_HEAD)
    sample_iters = list(range(0, min(30, len(history)), 5)) + [len(history) - 1]
    seen_iters: set = set()
    for it in sample_iters:
        if it in seen_iters or it >= len(history):
            continue
        seen_iters.add(it)
        snap = history[it]
        t.add_row(str(it + 1), *[f"{snap.get(nid, 0):.6f}" for nid in top3])
    console.print(t)

    # ── Takeaways ───────────────────────────────────────────────────────────
    console.print()
    console.print("[bold green]Key Influence Graph Takeaways:[/bold green]")
    console.print(
        "  • [cyan]Directed edges[/cyan]           — asymmetric influence: A→B ≠ B→A; "
        "two indexes cover both directions at O(log E + k)"
    )
    console.print(
        "  • [cyan]Weighted PageRank[/cyan]         — rank flows proportional to edge weight; "
        "a highly-rated destination accumulates more authority than a merely popular one"
    )
    console.print(
        "  • [cyan]Personalized PageRank[/cyan]     — concentrating teleport on a seed makes "
        "PageRank local to that seed's neighbourhood; no n×n matrix needed"
    )
    console.print(
        "  • [cyan]Random Walk with Restart[/cyan]  — Monte Carlo approximation of PPR; "
        "sub-linear runtime since only a small neighbourhood is visited"
    )
    console.print(
        "  • [cyan]HITS hub vs authority[/cyan]     — separates 'gateway' films from "
        "'destination' films; undirected degree misses this asymmetry entirely"
    )
    console.print(
        "  • [cyan]Convergence speed[/cyan]         — power iteration on small graphs "
        "converges in < 30 iterations; distributed Pregel (Spark GraphX, Beam) "
        "scales this to billions of nodes"
    )
    console.print(
        "  [dim]Production stacks: YouTube watch-graph (TensorFlow Graph Neural Net + "
        "PageRank signal), Pinterest Pixie (C++ RWR, microsecond latency), "
        "Google Knowledge Graph (entity authority for movie search results)[/dim]"
    )


if __name__ == "__main__":
    main()
