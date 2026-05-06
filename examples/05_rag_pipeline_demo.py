"""
Demo 5: RAG Pipeline
=====================
Shows document chunking, vector indexing, context retrieval, and prompt construction.
No LLM API key required — returns the built prompt ready for any LLM.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from databaseai.rag_pipeline import RAGIngestion, RAGRetrieval
from databaseai.seed_data import MOVIES

console = Console()


def main():
    console.rule("[bold cyan]RAG Pipeline Demo[/bold cyan]")

    # Ingest
    ingestion = RAGIngestion()
    docs = [
        {"id": m["id"], "title": m["title"], "content": m["description"],
         "metadata": {"genre": m["genre"], "year": m["year"]}}
        for m in MOVIES
    ]
    result = ingestion.ingest_batch(docs)
    console.print(f"\n[green]✓[/green] Ingested {result['documents_ingested']} documents "
                  f"→ {result['total_chunks']} chunks indexed in vector DB")
    console.print(f"  Chunk size: {RAGIngestion.CHUNK_SIZE} chars, "
                  f"overlap: {RAGIngestion.CHUNK_OVERLAP} chars")

    retrieval = RAGRetrieval.from_ingestion(ingestion)

    questions = [
        "What movies are about artificial intelligence or robots?",
        "Which films involve time travel or manipulation of time?",
        "What are the best films about crime and morality?",
        "Which movies feature animated worlds or fantasy realms?",
    ]

    for question in questions:
        console.print(f"\n[bold yellow]Q: {question}[/bold yellow]")

        chunks = retrieval.retrieve(question, n_chunks=3)
        t = Table("Similarity", "Title", "Excerpt", box=box.SIMPLE)
        for c in chunks:
            t.add_row(f"{c['similarity']:.4f}", c["title"], c["text"][:60] + "...")
        console.print(t)

    # Show full prompt for one question
    console.print()
    q = "What Christopher Nolan films are about the nature of reality?"
    prompt_result = retrieval.build_prompt(q, n_chunks=4)
    console.print(Panel(
        prompt_result["prompt"][:800] + "\n[dim]... (truncated)[/dim]",
        title=f'[bold]Built Prompt for: "{q}"[/bold]',
        subtitle=f"[dim]{prompt_result['context_char_count']} context chars, "
                 f"{len(prompt_result['context_chunks'])} chunks[/dim]",
        box=box.ROUNDED,
    ))
    console.print("\n[bold green]→ Pass this prompt to any LLM API for a grounded answer.[/bold green]")
    console.print("[dim]  No hallucination — the LLM can only answer from the retrieved context.[/dim]")


if __name__ == "__main__":
    main()
