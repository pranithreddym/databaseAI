"""
Directed Movie Influence Graph — SQLite-backed PageRank, HITS, and RWR
=======================================================================

Models movies as nodes in a directed weighted graph where an edge A→B means
"users who rated A also rated B".  Edge weight encodes how strongly co-viewers
endorsed B after experiencing A.

DB Architect notes:
  - Directed edges require only a single indexed SELECT per traversal direction:
    "WHERE from_node = ?" for outgoing links, "WHERE to_node = ?" for incoming.
    Two separate indexes (idx_from, idx_to) make both directions O(log E + k).
  - Edge weights are accumulated co-rating scores rather than raw counts.
    A highly-rated destination movie accumulates more incoming weight from every
    co-viewer, causing PageRank to flow toward genuinely acclaimed films.
  - PageRank is implemented as an iterative power iteration that terminates when
    the L1 delta between successive score vectors drops below a tolerance.
    Convergence typically occurs in 20-40 iterations for graphs this size.
  - Personalized PageRank (Random Walk with Restart) concentrates teleport mass
    on a seed node.  The resulting scores approximate how "related" each node is
    to the seed — proportional to the stationary distribution of a random walker
    that resets to the seed with probability restart_prob at each step.
  - HITS separates hub score (a movie that links to many high-authority movies)
    from authority score (a movie linked to by many high-hub movies).  In a
    recommendation graph, hubs are "gateway" films and authorities are
    "destination" films that people converge on from many different starting points.

Production parallels:
  - YouTube: watch-next graph built from co-viewed session pairs; a PageRank
    variant combined with neural ranking scores determines the algorithmic feed.
  - Pinterest Pixie: random walk with restart on a pin-board bipartite graph;
    the walk count approximates personalized PageRank without full matrix ops.
  - TikTok For You Page: short engagement graph edges from watch-time signals;
    distributed Pregel-style PageRank on billions of video nodes.
  - Google Knowledge Graph: entity authority scores determine which movie entity
    surfaces first in search results for ambiguous queries like "batman film".
"""

import random
import sqlite3
from collections import defaultdict
from contextlib import contextmanager


_DDL = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id   TEXT PRIMARY KEY,
    label     TEXT NOT NULL,
    genre     TEXT
);

CREATE TABLE IF NOT EXISTS edges (
    from_node TEXT NOT NULL REFERENCES nodes(node_id),
    to_node   TEXT NOT NULL REFERENCES nodes(node_id),
    weight    REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (from_node, to_node)
);

CREATE INDEX IF NOT EXISTS idx_inf_from ON edges (from_node);
CREATE INDEX IF NOT EXISTS idx_inf_to   ON edges (to_node);
"""


class InfluenceGraph:
    """
    Directed weighted graph of movie-to-movie influence.

    Real-world parallel:
      YouTube's co-watch graph — when many viewers watch B immediately after A,
      a directed edge A→B accumulates weight and PageRank flows from A to B,
      making B more likely to appear in the algorithmic "Up Next" feed.
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

    # ── Node / edge management ────────────────────────────────────────────────

    def add_node(self, node_id: str, label: str, genre: str = None) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO nodes (node_id, label, genre) VALUES (?, ?, ?)",
                (node_id, label, genre),
            )

    def add_edge(self, from_node: str, to_node: str, weight: float = 1.0) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO edges (from_node, to_node, weight) VALUES (?, ?, ?)",
                (from_node, to_node, weight),
            )

    def build_from_seed(self, movies: list, ratings: list) -> None:
        """
        Derive directed influence edges from co-rating data.

        For every pair (A, B) rated by the same user, add a directed edge A→B
        weighted by the destination movie's rating from that user.  A highly-rated
        B accumulates larger incoming weight, so PageRank flows preferentially
        toward critically acclaimed films — mirroring how YouTube's watch-time
        signal boosts high-retention videos in the recommendation graph.
        """
        for m in movies:
            self.add_node(m["id"], m["title"], m.get("genre"))

        user_ratings: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for row in ratings:
            user_id, movie_id, rating = row[0], row[1], float(row[2])
            user_ratings[user_id].append((movie_id, rating))

        # Accumulate A→B weights across all users
        accumulated: dict[tuple[str, str], float] = defaultdict(float)
        for rated in user_ratings.values():
            for m_a, _ in rated:
                for m_b, r_b in rated:
                    if m_a != m_b:
                        # Destination movie's rating drives edge weight
                        accumulated[(m_a, m_b)] += r_b

        for (from_node, to_node), weight in accumulated.items():
            self.add_edge(from_node, to_node, weight)

    # ── Graph statistics ──────────────────────────────────────────────────────

    def node_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

    def edge_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    def in_degree(self, node_id: str) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM edges WHERE to_node = ?", (node_id,)
            ).fetchone()[0]

    def out_degree(self, node_id: str) -> int:
        with self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM edges WHERE from_node = ?", (node_id,)
            ).fetchone()[0]

    def get_node(self, node_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT node_id, label, genre FROM nodes WHERE node_id = ?",
                (node_id,),
            ).fetchone()
        return dict(row) if row else None

    def degree_centrality(self) -> list[dict]:
        """All nodes sorted by total (in + out) degree descending."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT n.node_id, n.label, n.genre,
                          COALESCE(out_e.out_deg, 0) AS out_degree,
                          COALESCE(in_e.in_deg,  0) AS in_degree,
                          COALESCE(out_e.out_deg, 0) + COALESCE(in_e.in_deg, 0) AS total_degree
                   FROM nodes n
                   LEFT JOIN (SELECT from_node, COUNT(*) AS out_deg FROM edges GROUP BY from_node)
                             out_e ON n.node_id = out_e.from_node
                   LEFT JOIN (SELECT to_node, COUNT(*) AS in_deg FROM edges GROUP BY to_node)
                             in_e ON n.node_id = in_e.to_node
                   ORDER BY total_degree DESC""",
            ).fetchall()
        return [dict(r) for r in rows]

    def incoming_weight(self, node_id: str) -> float:
        """Sum of weights on all edges pointing to node_id."""
        with self._conn() as conn:
            result = conn.execute(
                "SELECT COALESCE(SUM(weight), 0.0) FROM edges WHERE to_node = ?",
                (node_id,),
            ).fetchone()[0]
        return float(result)

    def top_by_incoming_weight(self, n: int = 10) -> list[dict]:
        """Nodes ranked by total incoming edge weight (raw influence pressure)."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT nd.node_id, nd.label, nd.genre,
                          COALESCE(SUM(e.weight), 0.0) AS total_incoming_weight,
                          COUNT(e.from_node) AS in_degree
                   FROM nodes nd
                   LEFT JOIN edges e ON e.to_node = nd.node_id
                   GROUP BY nd.node_id
                   ORDER BY total_incoming_weight DESC
                   LIMIT ?""",
                (n,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Traversal helpers ─────────────────────────────────────────────────────

    def _all_node_ids(self) -> list[str]:
        with self._conn() as conn:
            return [r[0] for r in conn.execute(
                "SELECT node_id FROM nodes ORDER BY node_id"
            ).fetchall()]

    def _out_edges(self, node_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT to_node, weight FROM edges WHERE from_node = ?", (node_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def _in_edges(self, node_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT from_node, weight FROM edges WHERE to_node = ?", (node_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── PageRank ──────────────────────────────────────────────────────────────

    def pagerank(
        self,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
        personalized: dict | None = None,
    ) -> dict[str, float]:
        """
        Weighted PageRank with optional personalization.

        Teleport probability (1 - damping) is distributed according to the
        personalized dict when supplied, or uniformly otherwise.
        Supplying personalized={seed_id: 1.0} yields Personalized PageRank
        (Random Walk with Restart), which scores each node by its proximity
        to the seed — the algorithm behind Pinterest's Pixie recommender.
        """
        nodes = self._all_node_ids()
        n = len(nodes)
        if n == 0:
            return {}

        # Weighted out-degree for normalization
        with self._conn() as conn:
            raw = conn.execute(
                "SELECT from_node, SUM(weight) FROM edges GROUP BY from_node"
            ).fetchall()
        out_total: dict[str, float] = {r[0]: float(r[1]) for r in raw}

        ranks: dict[str, float] = {nd: 1.0 / n for nd in nodes}

        if personalized:
            total_p = sum(personalized.values())
            teleport: dict[str, float] = {k: v / total_p for k, v in personalized.items()}
        else:
            teleport = {nd: 1.0 / n for nd in nodes}

        for _ in range(max_iter):
            # Dangling nodes (no out-edges) leak rank; redistribute their share
            # uniformly — the standard dangling-node correction that makes scores
            # sum to 1.0 regardless of graph structure.
            dangling_sum = sum(
                ranks[nd] for nd in nodes if out_total.get(nd, 0.0) == 0.0
            )

            new_ranks: dict[str, float] = {}
            for nd in nodes:
                incoming = 0.0
                for edge in self._in_edges(nd):
                    src = edge["from_node"]
                    w = edge["weight"]
                    out_w = out_total.get(src, 0.0)
                    if out_w > 0:
                        incoming += ranks[src] * (w / out_w)
                # Add dangling-node correction: distribute leaked rank uniformly
                incoming += dangling_sum / n
                new_ranks[nd] = damping * incoming + (1.0 - damping) * teleport.get(nd, 0.0)

            delta = sum(abs(new_ranks[nd] - ranks[nd]) for nd in nodes)
            ranks = new_ranks
            if delta < tol:
                break

        return ranks

    def pagerank_history(
        self,
        damping: float = 0.85,
        max_iter: int = 50,
        track_nodes: list[str] | None = None,
    ) -> list[dict[str, float]]:
        """
        Run PageRank and return score snapshots at every iteration.
        Used to visualise convergence speed.
        """
        nodes = self._all_node_ids()
        n = len(nodes)
        if n == 0:
            return []

        with self._conn() as conn:
            raw = conn.execute(
                "SELECT from_node, SUM(weight) FROM edges GROUP BY from_node"
            ).fetchall()
        out_total = {r[0]: float(r[1]) for r in raw}
        teleport = 1.0 / n
        ranks: dict[str, float] = {nd: teleport for nd in nodes}
        tracked = track_nodes or nodes[:5]

        history: list[dict[str, float]] = []
        for _ in range(max_iter):
            dangling_sum = sum(
                ranks[nd] for nd in nodes if out_total.get(nd, 0.0) == 0.0
            )
            new_ranks: dict[str, float] = {}
            for nd in nodes:
                incoming = 0.0
                for edge in self._in_edges(nd):
                    src = edge["from_node"]
                    w = edge["weight"]
                    out_w = out_total.get(src, 0.0)
                    if out_w > 0:
                        incoming += ranks[src] * (w / out_w)
                incoming += dangling_sum / n
                new_ranks[nd] = damping * incoming + (1.0 - damping) * teleport
            ranks = new_ranks
            history.append({nd: round(ranks[nd], 6) for nd in tracked})

        return history

    # ── HITS ─────────────────────────────────────────────────────────────────

    def hits(
        self, max_iter: int = 100, tol: float = 1e-6
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        HITS — Hyperlink-Induced Topic Search.

        Hub score:       a high-hub movie leads viewers to many high-authority films.
        Authority score: a high-authority movie is referenced by many high-hub films.

        In the recommendation context:
          - High hub    → "gateway" film; watching it reliably steers users toward
                          acclaimed titles (e.g. an introductory sci-fi that leads
                          users to deeper, more celebrated works).
          - High authority → "destination" film; many viewing paths converge on it
                          regardless of starting point (the critically acclaimed title
                          everyone ends up watching eventually).
        """
        nodes = self._all_node_ids()
        hubs: dict[str, float] = {nd: 1.0 for nd in nodes}
        auths: dict[str, float] = {nd: 1.0 for nd in nodes}

        for _ in range(max_iter):
            # Authority update: sum hub scores of in-neighbors (weighted)
            new_auths: dict[str, float] = {nd: 0.0 for nd in nodes}
            for nd in nodes:
                for edge in self._in_edges(nd):
                    new_auths[nd] += hubs[edge["from_node"]] * edge["weight"]

            # Hub update: sum authority scores of out-neighbors (weighted)
            new_hubs: dict[str, float] = {nd: 0.0 for nd in nodes}
            for nd in nodes:
                for edge in self._out_edges(nd):
                    new_hubs[nd] += new_auths[edge["to_node"]] * edge["weight"]

            # L2 normalisation keeps scores from exploding
            auth_norm = (sum(v ** 2 for v in new_auths.values()) ** 0.5) or 1.0
            hub_norm  = (sum(v ** 2 for v in new_hubs.values())  ** 0.5) or 1.0
            new_auths = {nd: v / auth_norm for nd, v in new_auths.items()}
            new_hubs  = {nd: v / hub_norm  for nd, v in new_hubs.items()}

            delta = sum(abs(new_auths[nd] - auths[nd]) for nd in nodes)
            auths, hubs = new_auths, new_hubs
            if delta < tol:
                break

        return hubs, auths

    # ── Personalized PageRank / RWR ───────────────────────────────────────────

    def random_walk_with_restart(
        self,
        seed_node: str,
        steps: int = 5000,
        restart_prob: float = 0.15,
        rng_seed: int | None = None,
    ) -> dict[str, float]:
        """
        Simulate a random walk with restart from seed_node.

        At each step, with probability restart_prob the walker teleports back to
        seed_node; otherwise it follows a weighted random out-edge.  Visit-count
        fractions approximate the Personalized PageRank distribution — the
        algorithm used by Pinterest Pixie and YouTube for seeded recommendations.

        A deterministic rng_seed makes the simulation reproducible in tests.
        """
        nodes = self._all_node_ids()
        if not nodes or seed_node not in nodes:
            return {}

        rng = random.Random(rng_seed)
        visit_counts: dict[str, int] = defaultdict(int)
        current = seed_node

        for _ in range(steps):
            visit_counts[current] += 1
            out = self._out_edges(current)
            if not out or rng.random() < restart_prob:
                current = seed_node
            else:
                total_w = sum(e["weight"] for e in out)
                threshold = rng.random() * total_w
                cumulative = 0.0
                current = seed_node  # fallback
                for edge in out:
                    cumulative += edge["weight"]
                    if cumulative >= threshold:
                        current = edge["to_node"]
                        break

        total = sum(visit_counts.values())
        return {nd: visit_counts.get(nd, 0) / total for nd in nodes}

    # ── Ranking helper ────────────────────────────────────────────────────────

    def top_n_by_score(
        self, scores: dict[str, float], n: int = 10
    ) -> list[dict]:
        """Return top-N node dicts enriched with their score, sorted descending."""
        sorted_pairs = sorted(scores.items(), key=lambda x: -x[1])[:n]
        result = []
        for node_id, score in sorted_pairs:
            node = self.get_node(node_id)
            if node:
                node["score"] = round(score, 6)
                result.append(node)
        return result
