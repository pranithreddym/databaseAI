"""
CineAI Full End-to-End Demo
============================
Runs all five database layers in sequence to show how they work together.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from databaseai.seed_data import MOVIES, USERS, RATINGS
from databaseai.vector_db import MovieVectorStore
from databaseai.relational_db import MovieRegistry
from databaseai.nosql_db import UserSessionStore
from databaseai.feature_store import FeatureStore
from databaseai.rag_pipeline import RAGIngestion, RAGRetrieval

console = Console()


def section(title: str, subtitle: str = "") -> None:
    console.print()
    console.print(Panel(f"[bold cyan]{title}[/bold cyan]\n[dim]{subtitle}[/dim]", box=box.DOUBLE))


def main():
    console.print(Panel(
        "[bold white]CineAI — Databases in the AI World[/bold white]\n"
        "[dim]A Netflix-like movie platform built on 5 database types[/dim]",
        box=box.HEAVY, style="blue"
    ))

    # ------------------------------------------------------------------ #
    # 1. RELATIONAL DB                                                     #
    # ------------------------------------------------------------------ #
    section("1. Relational Database (SQLite)", "ACID-guaranteed structured storage")

    registry = MovieRegistry()
    registry.add_movies(MOVIES)
    for u in USERS:
        registry.add_user(u)
    for user_id, movie_id, score, review in RATINGS:
        registry.add_rating(user_id, movie_id, score, review)

    console.print(f"[green]✓[/green] Loaded {registry.movie_count()} movies, "
                  f"{registry.user_count()} users, {registry.rating_count()} ratings")

    top = registry.top_rated_movies(n=5)
    t = Table("Title", "Genre", "Avg Score", "Total Ratings", box=box.SIMPLE)
    for r in top:
        t.add_row(r["title"], r["genre"], str(r["avg_score"]), str(r["total_ratings"]))
    console.print("[bold]Top Rated Movies:[/bold]")
    console.print(t)

    # ------------------------------------------------------------------ #
    # 2. VECTOR DB                                                         #
    # ------------------------------------------------------------------ #
    section("2. Vector Database (ChromaDB)", "Semantic similarity search via embeddings")

    vector_store = MovieVectorStore()
    vector_store.upsert_movies(MOVIES)
    console.print(f"[green]✓[/green] Indexed {vector_store.count()} movies as embedding vectors")

    query = "astronauts travelling through space to save humanity"
    results = vector_store.find_similar(query, n=3)
    t = Table("Title", "Genre", "Year", "Similarity", box=box.SIMPLE)
    for r in results:
        t.add_row(r["title"], r["genre"], str(r["year"]), str(r["similarity_score"]))
    console.print(f'[bold]Semantic search: "{query}"[/bold]')
    console.print(t)

    # ------------------------------------------------------------------ #
    # 3. NOSQL / DOCUMENT DB                                               #
    # ------------------------------------------------------------------ #
    section("3. NoSQL Document Store", "Flexible schema for user sessions and events")

    sessions = UserSessionStore()
    sid = sessions.create_session("u01", {"type": "smart_tv", "model": "Samsung QN90B"})
    sessions.add_event("u01", sid, {"type": "search", "query": "sci-fi thriller"})
    sessions.add_event("u01", sid, {"type": "play", "movie_id": "m01"})
    sessions.add_event("u01", sid, {"type": "pause", "movie_id": "m01", "position_s": 3623})
    sessions.add_event("u01", sid, {"type": "resume", "movie_id": "m01", "position_s": 3623})
    sessions.add_event("u01", sid, {"type": "rate", "movie_id": "m01", "score": 5})
    sessions.end_session("u01", sid)

    doc = sessions.get_session("u01", sid)
    console.print(f"[green]✓[/green] Session {sid} — {len(doc['events'])} events")
    counts = sessions.get_event_counts("u01")
    console.print(f"  Event counts: {counts}")
    console.print(f"  Watch history: {sessions.get_watch_history('u01')}")

    # ------------------------------------------------------------------ #
    # 4. FEATURE STORE                                                     #
    # ------------------------------------------------------------------ #
    section("4. Feature Store", "Pre-computed ML features for training and serving")

    fs = FeatureStore()
    for u in USERS:
        user_ratings = registry.get_user_ratings(u["id"])
        fs.compute_user_features(u["id"], user_ratings, watch_count_7d=5, watch_count_30d=20)
    for m in MOVIES:
        stats = registry.movie_stats(m["id"])
        rlist = ([stats["avg_score"]] * int(stats["total_ratings"])
                 if stats["total_ratings"] else [])
        fs.compute_movie_features(m["id"], m["genre"], rlist)

    counts = fs.feature_count()
    console.print(f"[green]✓[/green] Computed {counts['user_features']} user feature vectors, "
                  f"{counts['movie_features']} movie feature vectors")

    uf = fs.get_user_features("u01")
    console.print(f"\n  User u01 features:")
    console.print(f"    avg_rating={uf['avg_rating']}, fav_genre={uf['fav_genre']}, "
                  f"watch_7d={uf['watch_count_7d']}")

    top_pop = fs.get_top_movies_by_popularity(n=3)
    console.print(f"\n  Top 3 by popularity score: "
                  f"{[r['movie_id'] for r in top_pop]}")

    # ------------------------------------------------------------------ #
    # 5. RAG PIPELINE                                                      #
    # ------------------------------------------------------------------ #
    section("5. RAG Pipeline (Vector DB → LLM Prompt)", "Retrieval-Augmented Generation")

    ingestion = RAGIngestion()
    docs = [
        {"id": m["id"], "title": m["title"], "content": m["description"],
         "metadata": {"genre": m["genre"], "year": m["year"]}}
        for m in MOVIES
    ]
    result = ingestion.ingest_batch(docs)
    console.print(f"[green]✓[/green] Ingested {result['documents_ingested']} documents → "
                  f"{result['total_chunks']} chunks indexed")

    retrieval = RAGRetrieval.from_ingestion(ingestion)
    questions = [
        "What movies involve dreams or alternate realities?",
        "Which films deal with artificial intelligence?",
        "What are the best crime thrillers?",
    ]

    for q in questions:
        answer = retrieval.answer_without_llm(q, n_chunks=3)
        console.print(f'\n[bold]Q: {q}[/bold]')
        console.print(f"  Retrieved {answer['retrieved_chunks']} chunks")
        for chunk in retrieval.retrieve(q, n_chunks=2):
            console.print(f"  → [{chunk['similarity']:.3f}] {chunk['title']}: {chunk['text'][:80]}...")

    # ------------------------------------------------------------------ #
    # SUMMARY                                                              #
    # ------------------------------------------------------------------ #
    section("Summary", "Each database type plays a distinct role")
    t = Table("Layer", "Technology", "Role in CineAI", box=box.SIMPLE_HEAD)
    t.add_row("Relational DB", "SQLite", "Movies, users, ratings — structured, ACID")
    t.add_row("Vector DB", "ChromaDB", "Semantic similarity search by plot embedding")
    t.add_row("NoSQL Doc DB", "JSON Store", "User sessions, event streams, flexible schema")
    t.add_row("Feature Store", "SQLite", "Pre-computed ML features, training/serving parity")
    t.add_row("RAG Pipeline", "ChromaDB + prompt", "Grounded LLM answers from retrieved context")
    console.print(t)
    console.print("\n[bold green]All 5 database layers operational.[/bold green]\n")


if __name__ == "__main__":
    main()
