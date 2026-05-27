"""
User-Movie Bipartite Graph — SQLite-backed collaborative filtering graph
========================================================================

Models users and movies as two disjoint node partitions connected by
weighted rating edges.  Two-hop traversal (user → rated movies → co-raters
→ their unrated movies) approximates item-based collaborative filtering
without materialising a full similarity matrix.

DB Architect notes:
  - A bipartite graph never has edges within the same partition (no user→user
    or movie→movie rows), which halves edge storage and lets neighbor lookups
    carry an implicit type filter.
  - Separate indexes on (from_node) and (to_node) make both traversal
    directions — "which movies did user X rate?" and "who rated movie M?" —
    equally fast O(log E + k) scans, equivalent to two hash-map adjacency
    lists but persistent and queryable in a single SQLite file.
  - Jaccard similarity (|A ∩ B| / |A ∪ B|) over rating sets is computed
    from two in-memory Python sets, each built with one indexed SELECT.
    At scale this is replaced by MinHash LSH (approximate, sub-linear) or
    ALS / BPR matrix factorisation (dense embedding space).
  - The 2-hop traversal is O(d_u × d_m × d_u') where d_u is user degree and
    d_m is movie degree.  For moderate catalogs (< 100 k items) this is fast
    enough in SQLite; production systems pre-compute and cache these scores.
  - Similarity-weighted scoring weights each candidate movie by the
    Jaccard similarity of the recommending co-rater, giving closer neighbours
    more influence than distant ones.

Production parallels:
  - Netflix implicit-feedback CF: user-movie edge weights are play-time
    fractions rather than explicit star ratings; ALS factorises the matrix.
  - Spotify "Discover Weekly": 2-hop walk on a user-playlist-song bipartite
    graph, then BPR re-ranks by listening history.
  - Pinterest PinSage: GNN on a pin-board bipartite graph; learned node
    embeddings replace hand-crafted Jaccard similarity.
  - Amazon "customers also bought": 2-hop bipartite walk aggregated daily
    and served from DynamoDB for sub-millisecond recommendation lookups.
"""

import sqlite3
from contextlib import contextmanager


_DDL = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id   TEXT PRIMARY KEY,
    node_type TEXT NOT NULL CHECK (node_type IN ('user', 'movie')),
    label     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    from_node TEXT NOT NULL REFERENCES nodes(node_id),
    to_node   TEXT NOT NULL REFERENCES nodes(node_id),
    weight    REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (from_node, to_node)
);

CREATE INDEX IF NOT EXISTS idx_edges_from ON edges (from_node);
CREATE INDEX IF NOT EXISTS idx_edges_to   ON edges (to_node);
"""


class CollabGraph:
    """
    Bipartite graph: User ↔ Movie edges weighted by rating score.

    Real-world parallel:
      The user-movie bipartite graph is the starting point for collaborative
      filtering — Netflix, Spotify, and Amazon all begin with this structure
      before adding matrix factorisation or neural layers on top.
    """

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        if db_path == ":memory:":
            self._shared = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared.row_factory = sqlite3.Row
        else:
            self._shared = None
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DDL)

    @contextmanager
    def _conn(self):
        if self._shared is not None:
            try:
                yield self._shared
                self._shared.commit()
            except Exception:
                self._shared.rollback()
                raise
        else:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # ── Node management ──────────────────────────────────────────────────────

    def add_user_node(self, user_id: str, label: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO nodes (node_id, node_type, label) VALUES (?, 'user', ?)",
                (user_id, label),
            )

    def add_movie_node(self, movie_id: str, label: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO nodes (node_id, node_type, label) VALUES (?, 'movie', ?)",
                (movie_id, label),
            )

    # ── Edge management ──────────────────────────────────────────────────────

    def add_rating_edge(self, user_id: str, movie_id: str, rating: float) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO edges (from_node, to_node, weight) VALUES (?, ?, ?)",
                (user_id, movie_id, rating),
            )

    def build_from_seed(self, movies: list, users: list, ratings: list) -> None:
        for u in users:
            self.add_user_node(u["id"], u["username"])
        for m in movies:
            self.add_movie_node(m["id"], m["title"])
        for row in ratings:
            user_id, movie_id, rating = row[0], row[1], row[2]
            self.add_rating_edge(user_id, movie_id, rating)

    # ── Traversal helpers ────────────────────────────────────────────────────

    def user_rated_movies(self, user_id: str) -> list[dict]:
        """Return movies rated by a user, sorted by rating descending."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT e.to_node AS movie_id, n.label AS title, e.weight AS rating
                   FROM edges e JOIN nodes n ON e.to_node = n.node_id
                   WHERE e.from_node = ? AND n.node_type = 'movie'
                   ORDER BY e.weight DESC""",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def movie_raters(self, movie_id: str) -> list[dict]:
        """Return users who rated a movie, sorted by rating descending."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT e.from_node AS user_id, n.label AS username, e.weight AS rating
                   FROM edges e JOIN nodes n ON e.from_node = n.node_id
                   WHERE e.to_node = ? AND n.node_type = 'user'
                   ORDER BY e.weight DESC""",
                (movie_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Recommendation algorithms ─────────────────────────────────────────────

    def collab_recommendations(self, user_id: str, top_n: int = 5) -> list[dict]:
        """
        2-hop traversal: user → rated movies → co-raters → their unrated movies.

        Ranks candidates by (co-rater vote count DESC, avg co-rater rating DESC).
        A movie that many neighbours loved ranks above one only a single neighbour liked.
        """
        rated = {r["movie_id"] for r in self.user_rated_movies(user_id)}
        if not rated:
            return []

        candidate_scores: dict[str, dict] = {}
        for movie_id in rated:
            for rater in self.movie_raters(movie_id):
                if rater["user_id"] == user_id:
                    continue
                for movie in self.user_rated_movies(rater["user_id"]):
                    mid = movie["movie_id"]
                    if mid not in rated:
                        if mid not in candidate_scores:
                            candidate_scores[mid] = {
                                "title": movie["title"],
                                "votes": 0,
                                "total": 0.0,
                            }
                        candidate_scores[mid]["votes"] += 1
                        candidate_scores[mid]["total"] += movie["rating"]

        results = [
            {
                "movie_id": mid,
                "title": s["title"],
                "co_rater_votes": s["votes"],
                "avg_rating": round(s["total"] / s["votes"], 2),
            }
            for mid, s in candidate_scores.items()
        ]
        results.sort(key=lambda x: (-x["co_rater_votes"], -x["avg_rating"]))
        return results[:top_n]

    def similarity_weighted_recommendations(self, user_id: str, top_n: int = 5) -> list[dict]:
        """
        Similarity-weighted score: Σ jaccard(user, co_rater) × co_rater_rating.

        Normalised by the sum of similarities so the scale stays near the
        original rating range.  Closer neighbours have greater influence than
        distant ones — the key difference from plain vote counting.
        """
        rated = {r["movie_id"] for r in self.user_rated_movies(user_id)}
        if not rated:
            return []

        sim_cache: dict[str, float] = {}
        for movie_id in rated:
            for rater in self.movie_raters(movie_id):
                rid = rater["user_id"]
                if rid != user_id and rid not in sim_cache:
                    sim_cache[rid] = self.user_similarity(user_id, rid)

        scores: dict[str, dict] = {}
        for co_rater_id, sim in sim_cache.items():
            if sim == 0.0:
                continue
            for movie in self.user_rated_movies(co_rater_id):
                mid = movie["movie_id"]
                if mid not in rated:
                    if mid not in scores:
                        scores[mid] = {"title": movie["title"], "num": 0.0, "denom": 0.0}
                    scores[mid]["num"] += sim * movie["rating"]
                    scores[mid]["denom"] += sim

        results = [
            {
                "movie_id": mid,
                "title": s["title"],
                "weighted_score": round(s["num"] / s["denom"], 3),
            }
            for mid, s in scores.items()
            if s["denom"] > 0
        ]
        results.sort(key=lambda x: -x["weighted_score"])
        return results[:top_n]

    # ── Similarity ────────────────────────────────────────────────────────────

    def user_similarity(self, user_a: str, user_b: str) -> float:
        """Jaccard similarity: |rated(A) ∩ rated(B)| / |rated(A) ∪ rated(B)|."""
        set_a = {r["movie_id"] for r in self.user_rated_movies(user_a)}
        set_b = {r["movie_id"] for r in self.user_rated_movies(user_b)}
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    def most_similar_users(self, user_id: str, top_n: int = 3) -> list[dict]:
        """Return the top-N users most similar to user_id by Jaccard score."""
        with self._conn() as conn:
            all_users = [
                r[0]
                for r in conn.execute(
                    "SELECT node_id FROM nodes WHERE node_type = 'user' AND node_id != ?",
                    (user_id,),
                ).fetchall()
            ]
        results = [
            {"user_id": uid, "similarity": self.user_similarity(user_id, uid)}
            for uid in all_users
        ]
        results.sort(key=lambda x: -x["similarity"])
        return results[:top_n]

    def similarity_matrix(self) -> list[dict]:
        """Return all pairwise Jaccard similarities for every user pair."""
        with self._conn() as conn:
            all_users = [
                r[0]
                for r in conn.execute(
                    "SELECT node_id FROM nodes WHERE node_type = 'user' ORDER BY node_id"
                ).fetchall()
            ]
        pairs = []
        for i in range(len(all_users)):
            for j in range(i + 1, len(all_users)):
                a, b = all_users[i], all_users[j]
                pairs.append(
                    {"user_a": a, "user_b": b, "similarity": self.user_similarity(a, b)}
                )
        pairs.sort(key=lambda x: -x["similarity"])
        return pairs

    # ── Popularity / statistics ───────────────────────────────────────────────

    def movie_popularity(self, top_n: int | None = None) -> list[dict]:
        """Movies ranked by number of distinct raters (in-degree), then avg rating."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT e.to_node AS movie_id, n.label AS title,
                          COUNT(e.from_node) AS rater_count,
                          ROUND(AVG(e.weight), 3) AS avg_rating
                   FROM edges e JOIN nodes n ON e.to_node = n.node_id
                   WHERE n.node_type = 'movie'
                   GROUP BY e.to_node
                   ORDER BY rater_count DESC, avg_rating DESC
                   LIMIT COALESCE(?, 9999999)""",
                (top_n,),
            ).fetchall()
        return [dict(r) for r in rows]

    def node_counts(self) -> dict:
        with self._conn() as conn:
            users = conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE node_type = 'user'"
            ).fetchone()[0]
            movies = conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE node_type = 'movie'"
            ).fetchone()[0]
            edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        return {"users": users, "movies": movies, "edges": edges}

    def user_labels(self) -> dict[str, str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT node_id, label FROM nodes WHERE node_type = 'user'"
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def movie_labels(self) -> dict[str, str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT node_id, label FROM nodes WHERE node_type = 'movie'"
            ).fetchall()
        return {r[0]: r[1] for r in rows}
