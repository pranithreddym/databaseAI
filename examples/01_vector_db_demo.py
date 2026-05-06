"""
Demo 1: Vector Database
========================
Shows how semantic similarity search works with movie plot embeddings.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.table import Table
from rich import box
from databaseai.vector_db import MovieVectorStore
from databaseai.seed_data import MOVIES

console = Console()


def main():
    console.rule("[bold cyan]Vector Database Demo[/bold cyan]")

    store = MovieVectorStore()
    store.upsert_movies(MOVIES)
    console.print(f"\n[green]✓[/green] Indexed {store.count()} movies\n")

    queries = [
        ("astronauts wormhole space-time relativity", None, None),
        ("dark hero fighting crime in a city", "action", None),
        ("animated magical spirit world", None, None),
        ("social thriller class inequality", None, 2015),
        ("mind bending alternate reality simulation", "sci-fi", None),
    ]

    for query, genre, year in queries:
        console.print(f'[bold yellow]Query:[/bold yellow] "{query}"'
                      + (f'  [dim]genre={genre}[/dim]' if genre else '')
                      + (f'  [dim]year≥{year}[/dim]' if year else ''))
        results = store.find_similar(query, n=4, genre_filter=genre, min_year=year)
        t = Table("Rank", "Title", "Genre", "Year", "Similarity", box=box.SIMPLE)
        for i, r in enumerate(results, 1):
            t.add_row(str(i), r["title"], r["genre"], str(r["year"]), f"{r['similarity_score']:.4f}")
        console.print(t)


if __name__ == "__main__":
    main()
