"""
Demo 16: Graph Database — User-Movie Bipartite Graph
=====================================================
Models users and movies as two disjoint node partitions connected by
weighted rating edges.  Demonstrates 2-hop collaborative filtering,
Jaccard user similarity, similarity-weighted recommendation scoring,
movie popularity via in-degree, and the cold-start problem.

Real-world parallel: Netflix / Spotify collaborative filtering — every
"Because you watched X" or "Discover Weekly" playlist starts with exactly
this bipartite graph before matrix factorisation or GNNs are layered on top.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from databaseai.collab_graph import CollabGraph
from databaseai.seed_data import MOVIES, USERS, RATINGS

console = Console()

_TITLE = {m["id"]: m["title"] for m in MOVIES}
_USER  = {u["id"]: u["username"] for u in USERS}


def _truncate(s: str, n: int = 28) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def main():
    console.rule("[bold cyan]Graph Database Demo 16 — User-Movie Bipartite Graph[/bold cyan]")
    console.print(
        "[dim]Real-world parallel: collaborative filtering — the bipartite graph "
        "behind Netflix 'Because you watched…' and Spotify Discover Weekly[/dim]\n"
    )

    g = CollabGraph()
    g.build_from_seed(MOVIES, USERS, RATINGS)
    counts = g.node_counts()

    # ── Section 1: Graph construction ─────────────────────────────────────────
    console.print(Panel(
        "[bold]1. Graph Construction — SQLite Bipartite Adjacency List[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        f"  [green]✓[/green] {counts['users']} user nodes  "
        f"| {counts['movies']} movie nodes  "
        f"| {counts['edges']} rating edges"
    )
    console.print(
        "  [dim]Schema: nodes(node_id, node_type∈{'user','movie'}, label) "
        "+ edges(from_node, to_node, weight)[/dim]"
    )
    console.print(
        "  [dim]Two indexes — idx_edges_from, idx_edges_to — make both "
        "'what did user X rate?' and 'who rated movie M?' O(log E + k)[/dim]"
    )

    # ── Section 2: User rating profiles ──────────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold]2. User Rating Profiles — Outgoing Edges per User[/bold]",
        box=box.ROUNDED,
    ))
    t = Table("User", "Rated Movies (sorted by rating)", box=box.SIMPLE_HEAD)
    for user in USERS:
        uid = user["id"]
        rated = g.user_rated_movies(uid)
        entries = [f"{_truncate(r['title'], 22)} ({r['rating']}★)" for r in rated]
        t.add_row(user["username"], "  ·  ".join(entries))
    console.print(t)

    # ── Section 3: 2-hop collaborative recommendations ───────────────────────
    console.print()
    console.print(Panel(
        "[bold]3. 2-Hop Collaborative Filtering — Vote-Count Ranked[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]Path: user → rated movies → co-raters → their unrated movies. "
        "Co-rater vote count is the rank signal — the more neighbours liked a "
        "candidate, the higher it surfaces.[/dim]"
    )
    t = Table("User", "Rank", "Recommendation", "Co-Rater Votes", "Avg Rating",
              box=box.SIMPLE_HEAD)
    for user in USERS:
        uid = user["id"]
        recs = g.collab_recommendations(uid, top_n=4)
        if not recs:
            t.add_row(user["username"], "—", "[dim]no recommendations (cold start)[/dim]", "", "")
            continue
        for i, rec in enumerate(recs, 1):
            t.add_row(
                user["username"] if i == 1 else "",
                str(i),
                _truncate(rec["title"]),
                str(rec["co_rater_votes"]),
                f"{rec['avg_rating']:.2f}★",
            )
    console.print(t)

    # ── Section 4: User-user Jaccard similarity matrix ────────────────────────
    console.print()
    console.print(Panel(
        "[bold]4. User-User Jaccard Similarity — |A ∩ B| / |A ∪ B|[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]Jaccard similarity over rating sets: two users who rated all the "
        "same movies score 1.0; no overlap scores 0.0.  At Netflix scale this is "
        "approximated with MinHash LSH in sub-linear time.[/dim]"
    )
    unames = [u["username"] for u in USERS]
    uids   = [u["id"]       for u in USERS]
    t = Table("", *unames, box=box.SIMPLE_HEAD)
    for i, (uid_a, uname_a) in enumerate(zip(uids, unames)):
        row = [uname_a]
        for uid_b in uids:
            if uid_a == uid_b:
                row.append("[bold]1.000[/bold]")
            else:
                sim = g.user_similarity(uid_a, uid_b)
                colour = "green" if sim > 0.2 else ("yellow" if sim > 0 else "dim")
                row.append(f"[{colour}]{sim:.3f}[/{colour}]")
        t.add_row(*row)
    console.print(t)

    # ── Section 5: Most similar user pairs ───────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold]5. Most Similar User Pairs — Top Neighbours per User[/bold]",
        box=box.ROUNDED,
    ))
    t = Table("User", "Most Similar", "2nd Similar", "3rd Similar", box=box.SIMPLE_HEAD)
    for user in USERS:
        uid = user["id"]
        similar = g.most_similar_users(uid, top_n=3)
        cells = [
            f"{_USER.get(s['user_id'], s['user_id'])} ({s['similarity']:.3f})"
            for s in similar
        ]
        while len(cells) < 3:
            cells.append("—")
        t.add_row(user["username"], *cells)
    console.print(t)

    # ── Section 6: Similarity-weighted recommendations ────────────────────────
    console.print()
    console.print(Panel(
        "[bold]6. Similarity-Weighted Recommendations — Closer Neighbours Win[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]Weighted score = Σ (jaccard(user, co_rater) × co_rater_rating) / Σ jaccard. "
        "A neighbour with Jaccard=0.5 has twice the influence of one with Jaccard=0.25. "
        "This is Pearson / cosine CF in the limit of dense ratings.[/dim]"
    )
    t = Table("User", "Rank", "Recommendation", "Weighted Score", box=box.SIMPLE_HEAD)
    for user in USERS:
        uid = user["id"]
        recs = g.similarity_weighted_recommendations(uid, top_n=4)
        if not recs:
            t.add_row(user["username"], "—", "[dim]cold start[/dim]", "")
            continue
        for i, rec in enumerate(recs, 1):
            t.add_row(
                user["username"] if i == 1 else "",
                str(i),
                _truncate(rec["title"]),
                f"{rec['weighted_score']:.3f}",
            )
    console.print(t)

    # ── Section 7: Movie popularity by in-degree ──────────────────────────────
    console.print()
    console.print(Panel(
        "[bold]7. Movie Popularity — In-Degree (Number of Raters)[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]In-degree = number of incoming rating edges.  High in-degree items "
        "get better collaborative signal; low in-degree items suffer from the "
        "long-tail sparsity problem that matrix factorisation addresses.[/dim]"
    )
    pop = g.movie_popularity(top_n=10)
    t = Table("Rank", "Movie", "Raters", "Avg Rating", "Coverage Note",
              box=box.SIMPLE_HEAD)
    for i, row in enumerate(pop, 1):
        note = (
            "[green]strong CF signal[/green]" if row["rater_count"] >= 3
            else ("[yellow]sparse[/yellow]" if row["rater_count"] == 2
                  else "[red]long tail — cold item[/red]")
        )
        t.add_row(
            str(i),
            _truncate(row["title"]),
            str(row["rater_count"]),
            f"{row['avg_rating']:.2f}★",
            note,
        )
    console.print(t)

    # ── Section 8: Cold-start problem ─────────────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold]8. The Cold-Start Problem — New User With No Ratings[/bold]",
        box=box.ROUNDED,
    ))
    console.print(
        "  [dim]A brand-new user has no outgoing edges, so the 2-hop traversal "
        "finds no co-raters and returns an empty list.  The three standard "
        "mitigations are shown below.[/dim]"
    )
    g.add_user_node("u_new", "new_user")
    cold_recs = g.collab_recommendations("u_new")
    console.print(
        f"  2-hop CF for new_user → [red]{len(cold_recs)} recommendations[/red] "
        f"(expected: 0)"
    )
    console.print()
    strategies = [
        ("Popularity fallback",
         "Serve the top-N most-rated items until the user accumulates ≥ k ratings."),
        ("Onboarding quiz",
         "Ask the user to rate 5–10 seed items; bootstrap the adjacency list immediately."),
        ("Content-based cold start",
         "Use item metadata (genre, director) to find candidate movies without any "
         "user ratings — exactly what Demo 08 (movie-to-movie graph) provides."),
    ]
    for i, (strategy, description) in enumerate(strategies, 1):
        console.print(f"  [cyan]{i}. {strategy}[/cyan]")
        console.print(f"     {description}")

    # ── Takeaways ──────────────────────────────────────────────────────────────
    console.print()
    console.print("[bold green]Key Bipartite Graph Takeaways:[/bold green]")
    console.print(
        "  • [cyan]Bipartite structure[/cyan]      — no intra-partition edges; "
        "two indexes cover both traversal directions at O(log E + k)"
    )
    console.print(
        "  • [cyan]2-hop CF[/cyan]                 — user → movie → co-rater → movie; "
        "O(d_u × d_m) per query, fast enough for in-process SQLite at demo scale"
    )
    console.print(
        "  • [cyan]Jaccard similarity[/cyan]        — |A ∩ B| / |A ∪ B| from two Python "
        "sets; MinHash LSH approximates this in O(k) for billion-user graphs"
    )
    console.print(
        "  • [cyan]Similarity-weighted score[/cyan] — closer neighbours influence "
        "recommendations more; approaches Pearson CF when ratings are dense"
    )
    console.print(
        "  • [cyan]Cold-start gap[/cyan]            — new nodes have zero edges; "
        "mitigated by popularity fallback, onboarding quizzes, or content-based seeding"
    )
    console.print(
        "  [dim]Production stacks: Netflix ALS on Spark, Spotify BPR + Word2Vec, "
        "Pinterest PinSage GNN, Amazon item-to-item CF served from DynamoDB[/dim]"
    )


if __name__ == "__main__":
    main()
