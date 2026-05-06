# databaseAI — Databases in the AI World

> A DB Architect's guide to every database type powering modern AI systems,
> demonstrated through a Netflix-like movie recommendation engine.

---

## Why This Project Exists

Modern AI systems are not powered by a single database. They use a carefully
orchestrated combination of database types, each chosen for a specific job.
This project explains each database type, why it exists in an AI stack, and
shows real working code.

---

## The Real-World Scenario

We are building **CineAI** — a Netflix-like movie recommendation platform.

```
User asks: "Recommend movies like Inception"
                        │
         ┌──────────────▼──────────────┐
         │        CineAI Backend        │
         └──────────────┬──────────────┘
                        │
     ┌──────────────────┼──────────────────┐
     │                  │                  │
     ▼                  ▼                  ▼
Vector DB           Relational DB      NoSQL DB
(ChromaDB)          (SQLite)           (JSON Store)
Find similar        Movie metadata     User sessions
movies by plot      Ratings, genres    Watch history
embedding           Users, reviews     Cache results
     │                  │                  │
     └──────────────────┼──────────────────┘
                        │
                        ▼
                  Feature Store
                   (SQLite)
              Pre-computed ML
              features for users
              and movies
                        │
                        ▼
                  RAG Pipeline
              Answer natural language
              questions about movies
```

---

## Database Types Covered

| # | Database Type | Technology Used | AI Use Case |
|---|---------------|-----------------|-------------|
| 1 | **Vector DB** | ChromaDB | Semantic search, similarity, RAG |
| 2 | **Relational DB** | SQLite | Metadata, structured queries, ACID |
| 3 | **NoSQL / Document DB** | JSON Store | Flexible schemas, user sessions |
| 4 | **Feature Store** | SQLite + Pandas | ML feature management, training data |
| 5 | **RAG Pipeline** | ChromaDB + custom | LLM context injection |

---

## Database Types: DB Architect Deep Dive

### 1. Vector Database — The AI-Native Database

**What it is:** Stores and searches high-dimensional numerical vectors (embeddings).

**Why AI needs it:**
- Text, images, audio are converted to vectors by neural networks
- Similarity search: "Find the 10 most similar items" (not exact match)
- Powers semantic search, recommendations, deduplication, RAG

**How it works:**
```
"The Dark Knight"  ──[embedding model]──▶  [0.23, -0.81, 0.44, ... 384 dims]
"Batman Begins"    ──[embedding model]──▶  [0.21, -0.79, 0.41, ... 384 dims]
"The Avengers"     ──[embedding model]──▶  [-0.12, 0.55, -0.33, ... 384 dims]

Query: "superhero crime thriller"
  ──[embedding model]──▶  [0.22, -0.80, 0.43, ...]
  ──[cosine similarity]──▶ Dark Knight (0.98), Batman Begins (0.95), Avengers (0.71)
```

**Indexing algorithms:** HNSW (Hierarchical Navigable Small World) — O(log n) search
**Real companies:** Pinecone, Weaviate, Qdrant, Milvus, pgvector, ChromaDB

---

### 2. Relational Database — The Trusted Foundation

**What it is:** Tables with rows/columns, SQL, ACID transactions.

**Why AI needs it:**
- Store structured metadata: movies, users, ratings, genres
- Model registry: track trained models, hyperparameters, metrics
- Experiment tracking: log every training run
- Audit trails: who changed what, when

**ACID guarantees matter:**
```
User rates a movie:
  BEGIN TRANSACTION
    INSERT INTO ratings (user_id, movie_id, score) VALUES (42, 101, 5)
    UPDATE users SET total_ratings = total_ratings + 1 WHERE id = 42
  COMMIT  -- Either both happen or neither does
```

---

### 3. NoSQL / Document Database — Flexible and Fast

**What it is:** Schema-less JSON document storage, horizontal scaling.

**Why AI needs it:**
- User interaction events (unpredictable shape)
- Model inference logs (vary per model type)
- A/B test configurations
- Cache for expensive LLM responses

**When SQL is wrong:**
```json
{
  "user_id": 42,
  "session": "2024-01-15T10:30:00",
  "events": [
    {"type": "search", "query": "sci-fi thriller", "results": 12},
    {"type": "play", "movie_id": 101, "position": 0},
    {"type": "pause", "movie_id": 101, "position": 1823},
    {"type": "rate", "movie_id": 101, "score": 5}
  ],
  "device": {"type": "smart_tv", "model": "Samsung QN90B"}
}
```
This nested, variable structure is painful in SQL. Documents handle it natively.

---

### 4. Feature Store — The ML Data Supply Chain

**What it is:** Centralized repository of pre-computed ML features with
point-in-time correctness.

**Why AI needs it:**
- Training/serving skew: ensure the same features are used in training AND inference
- Feature reuse: compute once, use across many models
- Backfill: reconstruct historical features for retraining
- Low-latency serving: pre-computed means microseconds, not seconds

**Offline vs Online store:**
```
Offline Store (historical):           Online Store (live):
  ┌─────────────────────────┐           ┌──────────────────────────┐
  │ user_id │ avg_rating │  │           │ Key: user:42             │
  │  42     │    4.2     │  │           │ avg_rating: 4.2          │
  │  43     │    3.8     │  │           │ fav_genre: sci-fi        │
  └─────────────────────────┘           │ watch_count_7d: 12       │
  Used for: model training              └──────────────────────────┘
                                        Used for: real-time inference
```

---

### 5. RAG Pipeline — Giving LLMs a Memory

**What it is:** Retrieval-Augmented Generation — inject relevant context
from a vector DB into an LLM prompt.

**Why AI needs it:**
- LLMs hallucinate facts — grounding them in a DB reduces this
- LLMs have a knowledge cutoff — RAG provides current data
- Cheaper than fine-tuning — update the DB, not the model

**How it works:**
```
User: "What movies did Christopher Nolan direct in the 2000s?"
         │
         ▼
[Embed the question]  →  query vector
         │
         ▼
[Search Vector DB]  →  top 5 most relevant movie docs
         │
         ▼
[Build prompt]:
  "Answer using ONLY this context:
   [Memento (2000) - directed by Nolan...]
   [Batman Begins (2005) - directed by Nolan...]
   [The Prestige (2006) - directed by Nolan...]
   Question: What movies did Christopher Nolan direct in the 2000s?"
         │
         ▼
[LLM answers based on retrieved facts]
```

---

## Project Structure

```
databaseAI/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Package config
├── .gitignore
├── docker-compose.yml           # Optional: run with real Redis/Postgres
│
├── src/databaseai/
│   ├── vector_db/               # ChromaDB vector search
│   ├── relational_db/           # SQLite metadata store
│   ├── nosql_db/                # JSON document store
│   ├── feature_store/           # ML feature management
│   └── rag_pipeline/            # RAG retrieval pipeline
│
├── tests/                       # pytest test suite
├── examples/                    # Runnable demo scripts
├── docs/                        # Architecture deep-dives
└── scripts/                     # Setup and run scripts
```

---

## Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/YOUR_USERNAME/databaseAI.git
cd databaseAI

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run all examples end-to-end
python examples/00_full_demo.py

# 5. Run tests
pytest tests/ -v

# 6. Run individual demos
python examples/01_vector_db_demo.py
python examples/02_relational_db_demo.py
python examples/03_nosql_demo.py
python examples/04_feature_store_demo.py
python examples/05_rag_pipeline_demo.py
```

---

## Architecture Decision Records

### ADR-001: Why ChromaDB over Pinecone?
ChromaDB is fully local, open-source, zero-config. Pinecone is a managed
cloud service. For learning and local dev, ChromaDB is the right choice.
In production, evaluate Pinecone/Weaviate/Qdrant based on scale needs.

### ADR-002: Why SQLite over PostgreSQL?
SQLite requires zero infrastructure. The relational concepts (joins,
transactions, indexes) are identical. Swap the connection string for
PostgreSQL in production.

### ADR-003: Why a custom JSON store over Redis/MongoDB?
Zero dependencies, zero infrastructure. MongoDB/Redis are the production
equivalents. The concepts are identical.

### ADR-004: Why not use a single database for everything?
This is the core lesson of this project. Each database is optimized for a
specific access pattern. Using one database for everything creates
performance cliffs and architectural dead ends.

---

## Performance Characteristics

| Database Type | Read Latency | Write Latency | Scale Pattern | Best For |
|---------------|-------------|---------------|---------------|----------|
| Vector DB | 1-50ms ANN | 10-100ms | Horizontal | Similarity search |
| Relational | <1ms indexed | <1ms | Vertical + read replicas | Structured queries |
| NoSQL Doc | <1ms | <1ms | Horizontal | Flexible schemas |
| Feature Store (online) | <1ms | async | Horizontal | Real-time inference |
| Feature Store (offline) | batch | batch | Distributed | Training |

---

## Author

Built as a DB Architect reference implementation.
Demonstrates production database patterns for AI systems.
