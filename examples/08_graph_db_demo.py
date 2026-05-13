"""
Demo 8: Graph Database
========================
Represents movies and their relationships (same director, same genre,
shared cast) as an adjacency-list graph stored in SQLite.
Demonstrates BFS/DFS traversal, degree centrality, and connected-component
discovery.

Real-world parallel: IMDb's "Six Degrees of Kevin Bacon" and Netflix's
"because you watched X" content-based recommendation chain.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from databaseai.seed_data import MOVIES
from databaseai.graph_db import MovieGraph, CAST

console = Console()


def _build_graph() -> MovieGraph:
    g = MovieGraph()
    g.build_from_seed(MOVIES, cast=CAST)
    return g


def _label(g, node_id):
    node = g.get_node(node_id)
    return node["label"] if node else node_id


def _path_str(g, path):
    return " → ".join(_label(g, nid) for nid in path)


def main():
    console.rule("[bold cyan]Graph Database Demo[/bold cyan]")
    console.print(
        "[dim]Real-world parallel: Six Degrees of Separation between movies "
        "— the adjacency-list pattern behind Netflix content graphs[/dim]\n"
    )

    g = _build_graph()

    console.print(Panel("[bold]1. Graph Construction — SQLite Adjacency List[/bold]", box=box.ROUNDED))
    etypes = g.edge_type_counts()
    t = Table("Edge Type", "Undirected Edges", "Meaning", box=box.SIMPLE_HEAD)
    descriptions = {
        "same_director": "Both films share the same director",
        "same_genre":    "Both films belong to the same genre",
        "co_actor":      "Both films share at least one cast member",
    }
    for etype, cnt in sorted(etypes.items()):
        t.add_row(etype, str(cnt), descriptions.get(etype, ""))
    console.print(t)
    console.print(
        f"  [green]✓[/green] {g.node_count()} movie nodes  "
        f"| {g.edge_count()} undirected edges  "
        f"({g.edge_count(directed=True)} directed rows in SQLite)\n"
        f"  [dim]Each undirected edge is stored as 2 directed rows so that "
        f"neighbor lookup is always a single SELECT WHERE from_node = ?[/dim]"
    )

    console.print()
    console.print(Panel("[bold]2. Adjacency List — Neighbors of \"Inception (2010)\"[/bold]", box=box.ROUNDED))
    inception_neighbors = g.neighbors("m01")
    seen_pairs: set = set()
    dedup = []
    for nb in inception_neighbors:
        key = (nb["node_id"], nb["edge_type"])
        if key not in seen_pairs:
            seen_pairs.add(key)
            dedup.append(nb)
    t = Table("Connected Movie", "Genre", "Edge Type", "Why", box=box.SIMPLE_HEAD)
    why_map = {
        "same_director": "Christopher Nolan",
        "same_genre":    "sci-fi",
        "co_actor":      "Michael Caine / Amy Adams / Scarlett Johansson",
    }
    for nb in dedup:
        t.add_row(nb["label"], nb["genre"] or "", nb["edge_type"], why_map.get(nb["edge_type"], ""))
    console.print(t)

    console.print()
    console.print(Panel("[bold]3. BFS — Shortest Paths (Six Degrees of Separation)[/bold]", box=box.ROUNDED))
    bfs_queries = [
        ("m01", "m10", "Inception → Mad Max: Fury Road"),
        ("m12", "m14", "Her → Knives Out"),
        ("m05", "m01", "Parasite → Inception"),
        ("m19", "m08", "12 Angry Men → Spirited Away (different cluster)"),
    ]
    t = Table("Query", "Hops", "Path", box=box.SIMPLE_HEAD)
    for src, dst, label in bfs_queries:
        path = g.bfs(src, dst)
        if path:
            t.add_row(label, str(len(path) - 1), _path_str(g, path))
        else:
            t.add_row(label, "[red]∞[/red]", "[red]No path — disconnected components[/red]")
    console.print(t)

    console.print()
    console.print(Panel("[bold]4. DFS vs BFS — Same Queries, Different Traversal Order[/bold]", box=box.ROUNDED))
    compare_pairs = [("m05", "m01", "Parasite → Inception"), ("m12", "m14", "Her → Knives Out")]
    t = Table("Query", "Algorithm", "Hops", "Path", box=box.SIMPLE_HEAD)
    for src, dst, label in compare_pairs:
        bfs_path = g.bfs(src, dst)
        dfs_path = g.dfs(src, dst)
        t.add_row(label, "BFS", str(len(bfs_path) - 1), _path_str(g, bfs_path))
        t.add_row("",     "DFS", str(len(dfs_path) - 1), _path_str(g, dfs_path))
    console.print(t)

    console.print()
    console.print(Panel("[bold]5. Degree Centrality — Most Connected Movies[/bold]", box=box.ROUNDED))
    top = g.most_connected(n=7)
    t = Table("Rank", "Movie", "Total Degree", box=box.SIMPLE_HEAD)
    for i, row in enumerate(top, 1):
        node = g.get_node(row["node_id"])
        genre = node["genre"] if node else ""
        t.add_row(str(i), f"{row['label']}  [dim]({genre})[/dim]", str(row["degree"]))
    console.print(t)

    console.print()
    console.print(Panel("[bold]6. Connected Components — Islands in the Movie Graph[/bold]", box=box.ROUNDED))
    components = g.connected_components()
    t = Table("Component", "Size", "Genre Mix", "Example Movie", box=box.SIMPLE_HEAD)
    for i, comp in enumerate(components, 1):
        genres = {g.get_node(nid)["genre"] for nid in comp if g.get_node(nid)}
        t.add_row(f"#{i}", str(len(comp)), ", ".join(sorted(genres)), _label(g, comp[0]))
    console.print(t)

    console.print()
    console.print(Panel("[bold]7. Edge-Type Filtered BFS — Director-Only Hops[/bold]", box=box.ROUNDED))
    director_pairs = [
        ("m01", "m02", "Inception → The Dark Knight (same director)"),
        ("m01", "m04", "Inception → The Matrix (different directors)"),
    ]
    t = Table("Query", "Edge Filter", "Path", box=box.SIMPLE_HEAD)
    for src, dst, label in director_pairs:
        path = g.bfs(src, dst, edge_type="same_director")
        t.add_row(label, "same_director", _path_str(g, path) if path else "[red]No path[/red]")
    console.print(t)

    console.print()
    console.print("[bold green]Key Graph DB Takeaways:[/bold green]")
    console.print("  • [cyan]Adjacency list[/cyan]         — (from_node, to_node, edge_type) rows with a composite index give O(log E + degree) neighbor lookup")
    console.print("  • [cyan]BFS[/cyan]                   — guarantees shortest path; use it for 'six degrees' and recommendation radius queries")
    console.print("  • [cyan]DFS[/cyan]                   — finds any path fast; use it for reachability checks and cycle detection")
    console.print("  • [cyan]Degree centrality[/cyan]     — high-degree hub films are pivots for serendipitous discovery")
    console.print("  • [cyan]Connected components[/cyan]  — genre islands reveal where co_actor / co_viewed edges are needed")
    console.print("  [dim]Production: Neo4j O(1) relationship traversal, Amazon Neptune (Gremlin/SPARQL), TigerGraph deep-link analytics[/dim]")


if __name__ == "__main__":
    main()
