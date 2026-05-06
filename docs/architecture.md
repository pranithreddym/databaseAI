# Architecture: Databases in the AI Stack

## The Core Insight

A production AI system is not one database — it is a pipeline where each
database type handles a job it is optimized for.

```
                           User Request
                               │
              ┌────────────────▼────────────────┐
              │          API Gateway             │
              └────────────────┬────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                     │
          ▼                    ▼                     ▼
    Feature Store        Vector DB            Relational DB
  (serve features       (ANN search           (metadata,
   in <1ms)             for similar            user data,
                         content)              ACID writes)
          │                    │                     │
          └────────────────────┼─────────────────────┘
                               │
                               ▼
                       NoSQL Event Store
                     (log every interaction
                      for analytics + retraining)
                               │
                               ▼
                        Offline Pipeline
                    (Spark/Flink batch jobs →
                     recompute features →
                     retrain models →
                     re-index vectors)
```

## Data Flow for "Recommend Movies Like Inception"

1. **Request arrives:** `GET /recommend?movie_id=m01&user_id=u01`
2. **Feature Store (online):** fetch user features in <1ms
   - `{avg_rating: 4.5, fav_genre: sci-fi, watch_count_7d: 5}`
3. **Vector DB:** find 20 most similar movies by plot embedding
   - Returns: Interstellar, The Matrix, Arrival, Her, ...
4. **Relational DB:** filter by user constraints
   - Exclude already watched, filter by preferred genres
5. **Ranking model:** score each candidate using user + movie features
6. **NoSQL Store:** log this recommendation event
   - `{type: "recommendation_shown", user_id: u01, movies: [...], algo_version: v2}`
7. **Response:** return ranked list

## Why Not Use One Database for Everything?

| Requirement | Vector DB | Relational | NoSQL | Feature Store |
|-------------|-----------|------------|-------|---------------|
| ANN search O(log n) | ✓ | ✗ (would need full scan) | ✗ | ✗ |
| ACID transactions | ✗ | ✓ | ✗ | ✗ |
| Flexible event schema | ✗ | ✗ | ✓ | ✗ |
| Sub-ms feature lookup | ✗ | Possible | Possible | ✓ |
| Point-in-time queries | ✗ | Limited | ✗ | ✓ |

Using a single database means sacrificing performance on at least 4 of 5 axes.

## Scaling Each Layer

### Vector DB
- **Small (< 1M vectors):** ChromaDB, local FAISS
- **Medium (< 100M):** Pinecone, Qdrant, Weaviate (managed)
- **Large (> 1B):** Milvus (distributed), custom HNSW on object storage

### Relational DB
- **Small:** SQLite
- **Medium:** PostgreSQL with read replicas
- **Large:** PostgreSQL + Citus (sharding), or CockroachDB (distributed)
- **Analytics:** Redshift, BigQuery (columnar OLAP)

### NoSQL Event Store
- **Small:** JSON files, TinyDB
- **Medium:** MongoDB, DynamoDB
- **Large:** Cassandra (write-optimized), Kafka + S3 (event stream)

### Feature Store
- **Small:** SQLite + Pandas
- **Medium:** Feast + Redis (online) + Parquet (offline)
- **Large:** Tecton, Hopsworks, SageMaker Feature Store

## The Feature Store in Detail

The feature store solves the **training/serving skew problem**:

```
Without Feature Store:
  Training pipeline:  avg_rating = sum(ratings) / len(ratings)   ← one formula
  Serving code:       avg_rating = sum(last_10) / 10             ← different formula
  Result: model trained on X but served Y → silent accuracy drop

With Feature Store:
  Both training and serving call:
    feature_store.get_user_features(user_id)
  Same features, same logic, guaranteed consistency.
```

## HNSW: How Vector Search Works at Scale

Hierarchical Navigable Small World (HNSW) builds a layered graph:

```
Layer 2 (sparse):   A ─────────────── Z
                     │                 │
Layer 1 (medium):   A ──── M ──────── Z
                     │      │          │
Layer 0 (dense):    A ─ B ─ C ─ ... ─ Z   ← all vectors connected to neighbors
```

Search navigates from sparse to dense, pruning the search space at each layer.
Result: O(log n) average search time vs O(n) brute-force.

ChromaDB uses HNSW internally. Pinecone and Weaviate use variants of it.
