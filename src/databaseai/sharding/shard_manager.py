"""
Sharding — consistent-hashing ring with SQLite shard backends
=============================================================

DB Architect notes:
  Vertical scaling (bigger machine) has a hard ceiling.  Horizontal sharding
  distributes rows across N independent database nodes, each owning a fraction
  of the keyspace.  The routing challenge is deciding which node owns a given
  key while keeping routing deterministic for both writes and reads.

  Naive modulo sharding (shard = hash(key) % N) breaks badly on resize: adding
  a single node invalidates N/(N+1) of all keys, forcing a near-total migration.

  Consistent hashing fixes this by mapping both nodes AND keys onto a circular
  ring (integer space 0 … 2^32-1).  Each key is assigned to the first node
  clockwise from its hash position.  Adding a node only steals keys from its
  clockwise neighbour; all other shards are untouched:
    - Expected migration fraction when adding the k-th node: 1/(k+1)
    - Adding shard 4 to a 3-shard ring migrates ≈ 25 % of keys, not 75 %.

  Virtual nodes (vnodes):
    A real ring with 3 physical nodes places 3 points, causing uneven key
    coverage.  Multiplying each node into V virtual nodes (each hashed with a
    different seed) distributes the arc lengths more uniformly.  Cassandra uses
    V ≈ 256 per node; DynamoDB uses a similar concept internally.

  Reference partition strategies:
    - Movies table replicated to every shard (small reference data, avoids
      cross-shard JOINs for recommendation queries).
    - Ratings table partitioned by user_id hash (a user's entire history lives
      on one shard, enabling fast per-user aggregations without scatter-gather).

  This module uses SQLite files as shard backends so the demo runs locally
  without any network infrastructure.  The routing, rebalancing, and fan-out
  logic is identical to what a production driver does against Cassandra nodes
  or DynamoDB partitions.

Production parallels:
  - Apache Cassandra: consistent hashing with Murmur3, configurable vnodes.
  - Amazon DynamoDB: transparent hash-key-based partitioning; rebalances
    automatically when a partition exceeds 10 GB or 3000 RCUs.
  - Vitess (MySQL sharding): range-based and hash-based sharding for YouTube,
    Slack, and GitHub.
  - CockroachDB: range-based, auto-rebalanced across nodes.
"""

import hashlib
import sqlite3
import os
import bisect
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Consistent Hash Ring
# ---------------------------------------------------------------------------

def _hash_key(value: str) -> int:
    """Map an arbitrary string onto the 32-bit ring uniformly via MD5."""
    return int(hashlib.md5(value.encode(), usedforsecurity=False).hexdigest(), 16) % (2 ** 32)


class ConsistentHashRing:
    """
    Circular ring over the 32-bit integer space.

    Each physical node is expanded into vnodes_per_node virtual points, each
    hashed as "<node_id>#<replica_index>".  Lookup walks the sorted ring to
    find the first virtual node whose position is >= the key's hash.

    Adding a node inserts vnodes_per_node new points; only keys that fall in
    the arcs now owned by the new node need to migrate from their prior owner.
    """

    def __init__(self, vnodes_per_node: int = 150) -> None:
        self._vnodes_per_node = vnodes_per_node
        self._ring: Dict[int, str] = {}      # position → node_id
        self._sorted_positions: List[int] = []

    @property
    def nodes(self) -> List[str]:
        return sorted(set(self._ring.values()))

    def add_node(self, node_id: str) -> None:
        for i in range(self._vnodes_per_node):
            pos = _hash_key(f"{node_id}#vn{i}")
            self._ring[pos] = node_id
        self._sorted_positions = sorted(self._ring.keys())

    def remove_node(self, node_id: str) -> None:
        keys_to_remove = [p for p, n in self._ring.items() if n == node_id]
        for k in keys_to_remove:
            del self._ring[k]
        self._sorted_positions = sorted(self._ring.keys())

    def get_node(self, key: str) -> Optional[str]:
        """Return the node responsible for key, or None if the ring is empty."""
        if not self._sorted_positions:
            return None
        h = _hash_key(key)
        idx = bisect.bisect_left(self._sorted_positions, h)
        if idx == len(self._sorted_positions):
            idx = 0
        return self._ring[self._sorted_positions[idx]]

    def key_distribution(self, keys: List[str]) -> Dict[str, int]:
        """Count how many of the given keys map to each node."""
        counts: Dict[str, int] = {n: 0 for n in self.nodes}
        for k in keys:
            node = self.get_node(k)
            if node:
                counts[node] = counts.get(node, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# SQLite shard helpers
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS movies (
    id       TEXT PRIMARY KEY,
    title    TEXT NOT NULL,
    genre    TEXT,
    year     INTEGER,
    director TEXT
);

CREATE TABLE IF NOT EXISTS ratings (
    user_id  TEXT NOT NULL,
    movie_id TEXT NOT NULL REFERENCES movies(id),
    score    REAL NOT NULL CHECK (score BETWEEN 0.0 AND 5.0),
    review   TEXT,
    PRIMARY KEY (user_id, movie_id)
);

CREATE INDEX IF NOT EXISTS idx_ratings_user  ON ratings(user_id);
CREATE INDEX IF NOT EXISTS idx_ratings_movie ON ratings(movie_id);
"""


def _open_shard(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_DDL)
    return conn


# ---------------------------------------------------------------------------
# Shard Manager
# ---------------------------------------------------------------------------

class ShardManager:
    """
    Routes reads and writes to the correct SQLite shard via a ConsistentHashRing.

    Movies are replicated to every shard (broadcast inserts).  Ratings are
    partitioned by user_id so all of a user's activity lives on one shard —
    this makes per-user recommendation queries shard-local and avoids expensive
    cross-shard scatter-gather for the common case.

    When a new shard is added via add_shard(), the manager migrates only the
    ratings whose user_id now hashes to the new node (the affected arc).  This
    is the critical property of consistent hashing: unaffected shards are never
    touched during rebalancing.
    """

    def __init__(self, shard_paths: List[str], vnodes_per_node: int = 150) -> None:
        self._ring = ConsistentHashRing(vnodes_per_node=vnodes_per_node)
        self._conns: Dict[str, sqlite3.Connection] = {}

        for path in shard_paths:
            node_id = self._node_id(path)
            self._conns[node_id] = _open_shard(path)
            self._ring.add_node(node_id)

    @staticmethod
    def _node_id(path: str) -> str:
        return os.path.basename(path)

    @property
    def shard_ids(self) -> List[str]:
        return self._ring.nodes

    def get_shard_for_user(self, user_id: str) -> str:
        """Return the shard node_id that owns user_id."""
        node = self._ring.get_node(user_id)
        if node is None:
            raise RuntimeError("No shards in ring")
        return node

    def _conn(self, node_id: str) -> sqlite3.Connection:
        return self._conns[node_id]

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def insert_movies(self, movies: List[dict]) -> None:
        """Replicate movies to every shard (reference data broadcast)."""
        for conn in self._conns.values():
            conn.executemany(
                "INSERT OR IGNORE INTO movies (id, title, genre, year, director) "
                "VALUES (:id, :title, :genre, :year, :director)",
                movies,
            )
            conn.commit()

    def insert_rating(self, user_id: str, movie_id: str,
                      score: float, review: str = "") -> str:
        """Route a rating to the shard that owns user_id; return node_id."""
        node = self.get_shard_for_user(user_id)
        self._conn(node).execute(
            "INSERT OR IGNORE INTO ratings (user_id, movie_id, score, review) "
            "VALUES (?, ?, ?, ?)",
            (user_id, movie_id, score, review),
        )
        self._conn(node).commit()
        return node

    def insert_ratings_bulk(self, ratings: List[Tuple]) -> Dict[str, int]:
        """
        Bulk-insert (user_id, movie_id, score, review) tuples, routing each
        by user_id.  Returns a per-shard count of rows inserted.
        """
        counts: Dict[str, int] = {n: 0 for n in self.shard_ids}
        for user_id, movie_id, score, review in ratings:
            node = self.insert_rating(user_id, movie_id, score, review)
            counts[node] += 1
        return counts

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def user_ratings(self, user_id: str) -> List[dict]:
        """Fetch all ratings for a single user — single-shard lookup."""
        node = self.get_shard_for_user(user_id)
        rows = self._conn(node).execute(
            """SELECT r.user_id, m.title, m.genre, r.score, r.review
               FROM ratings r JOIN movies m ON m.id = r.movie_id
               WHERE r.user_id = ?
               ORDER BY r.score DESC""",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def global_top_rated(self, limit: int = 5) -> List[dict]:
        """
        Fan-out: query every shard and merge results in the coordinator.
        Equivalent to a Cassandra scatter-gather or DynamoDB parallel scan.
        """
        totals: Dict[str, Tuple[float, int]] = {}   # movie_id → (sum_score, count)
        movie_meta: Dict[str, dict] = {}

        for conn in self._conns.values():
            rows = conn.execute(
                """SELECT m.id, m.title, m.genre, r.score
                   FROM ratings r JOIN movies m ON m.id = r.movie_id"""
            ).fetchall()
            for row in rows:
                mid = row["id"]
                s, c = totals.get(mid, (0.0, 0))
                totals[mid] = (s + row["score"], c + 1)
                movie_meta[mid] = {"title": row["title"], "genre": row["genre"]}

        results = []
        for mid, (total, count) in totals.items():
            results.append({
                "movie_id": mid,
                "title": movie_meta[mid]["title"],
                "genre": movie_meta[mid]["genre"],
                "avg_score": round(total / count, 2),
                "votes": count,
            })
        results.sort(key=lambda r: (-r["avg_score"], -r["votes"]))
        return results[:limit]

    def rating_count_per_shard(self) -> Dict[str, int]:
        counts = {}
        for node, conn in self._conns.items():
            row = conn.execute("SELECT COUNT(*) AS n FROM ratings").fetchone()
            counts[node] = row["n"]
        return counts

    def total_ratings(self) -> int:
        return sum(self.rating_count_per_shard().values())

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def add_shard(self, path: str) -> Tuple[str, int]:
        """
        Add a new shard to the ring and migrate the ratings that now belong to
        it.  Returns (new_node_id, rows_migrated).

        Only ratings whose user_id maps to the new node after re-hashing need
        to move.  The expected fraction is 1/(N+1) of the total keyspace — the
        consistent-hashing guarantee.
        """
        new_node = self._node_id(path)
        self._conns[new_node] = _open_shard(path)

        # Replicate movies to the new shard before inserting ratings.
        first_node = next(iter(self._conns.keys()))
        if first_node != new_node:
            movies = self._conns[first_node].execute(
                "SELECT id, title, genre, year, director FROM movies"
            ).fetchall()
            self._conns[new_node].executemany(
                "INSERT OR IGNORE INTO movies (id, title, genre, year, director) "
                "VALUES (?, ?, ?, ?, ?)",
                [(r["id"], r["title"], r["genre"], r["year"], r["director"])
                 for r in movies],
            )
            self._conns[new_node].commit()

        # Add to ring AFTER fetching old assignments so we can detect changes.
        old_assignments: Dict[str, str] = {}
        for existing_node, conn in self._conns.items():
            if existing_node == new_node:
                continue
            rows = conn.execute("SELECT DISTINCT user_id FROM ratings").fetchall()
            for row in rows:
                uid = row["user_id"]
                old_assignments[uid] = existing_node

        self._ring.add_node(new_node)

        # Find users whose home shard changed.
        migrated = 0
        for user_id, old_node in old_assignments.items():
            new_home = self._ring.get_node(user_id)
            if new_home == new_node:
                rows = self._conns[old_node].execute(
                    "SELECT user_id, movie_id, score, review FROM ratings "
                    "WHERE user_id = ?",
                    (user_id,),
                ).fetchall()
                self._conns[new_node].executemany(
                    "INSERT OR IGNORE INTO ratings (user_id, movie_id, score, review) "
                    "VALUES (?, ?, ?, ?)",
                    [(r["user_id"], r["movie_id"], r["score"], r["review"])
                     for r in rows],
                )
                self._conns[new_node].commit()
                self._conns[old_node].execute(
                    "DELETE FROM ratings WHERE user_id = ?", (user_id,)
                )
                self._conns[old_node].commit()
                migrated += len(rows)

        return new_node, migrated

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        for conn in self._conns.values():
            conn.close()
        self._conns.clear()
