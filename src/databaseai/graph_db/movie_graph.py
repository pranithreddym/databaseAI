"""
Movie Graph — SQLite-backed adjacency list
==========================================

Represents movies as graph nodes and their relationships
(same director, same genre, shared cast) as weighted directed edges.

DB Architect notes:
  - An adjacency list encodes a graph as rows: (from_node, to_node, edge_type).
    This is the natural relational representation — no special graph engine needed
    for graphs that fit in memory or at moderate scale.
  - The index on (from_node, edge_type) turns a neighbor lookup from a full
    table scan (O(E)) into O(log E + degree) — identical to a hash-map adjacency
    list but persistent and query-able with SQL.
  - Edges are stored bidirectionally so that "SELECT to_node WHERE from_node = ?"
    is all we ever need; no UNION required.
  - BFS guarantees the shortest path (minimum hop count). DFS finds *any* path
    quickly but may return a longer route. The choice depends on the query:
    "six degrees of separation" needs BFS; "does any path exist?" can use DFS.
  - Connected-component discovery runs BFS over all unvisited nodes: O(V + E).
  - For production graphs (billions of edges): Neo4j stores nodes and
    relationships as linked-list records enabling O(1) neighbor traversal;
    Amazon Neptune speaks Gremlin or SPARQL; TigerGraph specialises in
    deep multi-hop analytics (>3 hops) used for fraud detection.

Production parallels:
  - Netflix "because you watched X" traverses a movie-genre-director bipartite
    graph finding titles 1-2 hops away from the seed title.
  - LinkedIn "People You May Know" runs BFS capped at depth 2, then ranks
    candidates by mutual-connection count.
  - IMDb's Six Degrees of Kevin Bacon is BFS over the actor-movie bipartite
    graph; the average path length is ~2.9 hops.
"""

import sqlite3
from collections import deque
from contextlib import contextmanager
from typing import Optional


CAST: dict[str, list[str]] = {
    "m01": ["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Michael Caine"],
    "m02": ["Christian Bale", "Heath Ledger", "Michael Caine"],
    "m03": ["Matthew McConaughey", "Anne Hathaway", "Michael Caine"],
    "m04": ["Keanu Reeves", "Laurence Fishburne", "Carrie-Anne Moss"],
    "m05": ["Song Kang-ho", "Lee Sun-kyun", "Cho Yeo-jeong"],
    "m06": ["John Travolta", "Uma Thurman", "Samuel L. Jackson"],
    "m07": ["Tim Robbins", "Morgan Freeman", "William Sadler"],
    "m08": ["Daveigh Chase", "Suzanne Pleshette", "Miyu Irino"],
    "m09": ["Daniel Kaluuya", "Allison Williams", "Bradley Whitford"],
    "m10": ["Tom Hardy", "Charlize Theron", "Nicholas Hoult"],
    "m11": ["Amy Adams", "Jeremy Renner", "Forest Whitaker"],
    "m12": ["Joaquin Phoenix", "Scarlett Johansson", "Amy Adams"],
    "m13": ["Anthony Gonzalez", "Gael Garcia Bernal", "Benjamin Bratt"],
    "m14": ["Daniel Craig", "Chris Evans", "Ana de Armas"],
    "m15": ["Marlon Brando", "Al Pacino", "James Caan"],
    "m16": ["Michelle Yeoh", "Stephanie Hsu", "Ke Huy Quan"],
    "m17": ["Toni Collette", "Milly Shapiro", "Gabriel Byrne"],
    "m18": ["Ryan Gosling", "Ana de Armas", "Harrison Ford"],
    "m19": ["Henry Fonda", "Lee J. Cobb", "Ed Begley"],
    "m20": ["Robert Downey Jr.", "Chris Evans", "Scarlett Johansson"],
}


DDL = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id  TEXT PRIMARY KEY,
    label    TEXT NOT NULL,
    genre    TEXT,
    director TEXT,
    year     INTEGER
);

CREATE TABLE IF NOT EXISTS edges (
    from_node TEXT NOT NULL REFERENCES nodes(node_id),
    to_node   TEXT NOT NULL REFERENCES nodes(node_id),
    edge_type TEXT NOT NULL,
    weight    REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (from_node, to_node, edge_type)
);

CREATE INDEX IF NOT EXISTS idx_edges_from
    ON edges (from_node, edge_type);

CREATE INDEX IF NOT EXISTS idx_edges_to
    ON edges (to_node, edge_type);
"""


class MovieGraph:
    """
    Graph of movies connected by director, genre, and shared cast.

    Real-world parallel:
      A recommendation engine walks this graph to answer
      "find me movies related to X within 2 hops" — the backbone of
      Netflix's content-based filtering before collaborative signals
      are layered on top.
    """

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        if db_path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_conn.row_factory = sqlite3.Row
        else:
            self._shared_conn = None
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(DDL)

    @contextmanager
    def _conn(self):
        if self._shared_conn is not None:
            try:
                yield self._shared_conn
                self._shared_conn.commit()
            except Exception:
                self._shared_conn.rollback()
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

    def add_node(self, node_id, label, genre=None, director=None, year=None):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO nodes (node_id, label, genre, director, year) VALUES (?, ?, ?, ?, ?)",
                (node_id, label, genre, director, year),
            )

    def add_edge(self, from_node, to_node, edge_type, weight=1.0):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO edges (from_node, to_node, edge_type, weight) VALUES (?, ?, ?, ?)",
                (from_node, to_node, edge_type, weight),
            )
            conn.execute(
                "INSERT OR IGNORE INTO edges (from_node, to_node, edge_type, weight) VALUES (?, ?, ?, ?)",
                (to_node, from_node, edge_type, weight),
            )

    def build_from_seed(self, movies, cast=None):
        cast = cast or {}
        for m in movies:
            self.add_node(m["id"], f"{m['title']} ({m['year']})",
                          genre=m.get("genre"), director=m.get("director"), year=m.get("year"))
        by_director: dict[str, list] = {}
        by_genre:    dict[str, list] = {}
        by_actor:    dict[str, list] = {}
        for m in movies:
            mid = m["id"]
            if m.get("director"):
                by_director.setdefault(m["director"], []).append(mid)
            if m.get("genre"):
                by_genre.setdefault(m["genre"], []).append(mid)
            for actor in cast.get(mid, []):
                by_actor.setdefault(actor, []).append(mid)

        def _pair_edges(groups, etype):
            for members in groups.values():
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        self.add_edge(members[i], members[j], etype)

        _pair_edges(by_director, "same_director")
        _pair_edges(by_genre,    "same_genre")
        _pair_edges(by_actor,    "co_actor")

    def get_node(self, node_id):
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM nodes WHERE node_id = ?", (node_id,)).fetchone()
        return dict(row) if row else None

    def neighbors(self, node_id, edge_type=None):
        with self._conn() as conn:
            if edge_type:
                rows = conn.execute(
                    """SELECT e.to_node AS node_id, e.edge_type, e.weight,
                              n.label, n.genre, n.director
                       FROM edges e JOIN nodes n ON e.to_node = n.node_id
                       WHERE e.from_node = ? AND e.edge_type = ?
                       ORDER BY n.label""",
                    (node_id, edge_type),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT e.to_node AS node_id, e.edge_type, e.weight,
                              n.label, n.genre, n.director
                       FROM edges e JOIN nodes n ON e.to_node = n.node_id
                       WHERE e.from_node = ?
                       ORDER BY e.edge_type, n.label""",
                    (node_id,),
                ).fetchall()
        return [dict(r) for r in rows]

    def degree(self, node_id, edge_type=None):
        with self._conn() as conn:
            if edge_type:
                return conn.execute(
                    "SELECT COUNT(DISTINCT to_node) FROM edges WHERE from_node = ? AND edge_type = ?",
                    (node_id, edge_type),
                ).fetchone()[0]
            return conn.execute(
                "SELECT COUNT(DISTINCT to_node) FROM edges WHERE from_node = ?",
                (node_id,),
            ).fetchone()[0]

    def most_connected(self, n=5, edge_type=None):
        with self._conn() as conn:
            if edge_type:
                rows = conn.execute(
                    """SELECT e.from_node AS node_id, nd.label,
                              COUNT(DISTINCT e.to_node) AS degree
                       FROM edges e JOIN nodes nd ON e.from_node = nd.node_id
                       WHERE e.edge_type = ?
                       GROUP BY e.from_node ORDER BY degree DESC LIMIT ?""",
                    (edge_type, n),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT e.from_node AS node_id, nd.label,
                              COUNT(DISTINCT e.to_node) AS degree
                       FROM edges e JOIN nodes nd ON e.from_node = nd.node_id
                       GROUP BY e.from_node ORDER BY degree DESC LIMIT ?""",
                    (n,),
                ).fetchall()
        return [dict(r) for r in rows]

    def _load_adj(self, edge_type=None):
        with self._conn() as conn:
            if edge_type:
                rows = conn.execute(
                    "SELECT from_node, to_node FROM edges WHERE edge_type = ?", (edge_type,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT DISTINCT from_node, to_node FROM edges").fetchall()
        adj: dict[str, list] = {}
        for from_n, to_n in rows:
            adj.setdefault(from_n, []).append(to_n)
        return adj

    def bfs(self, start_id, end_id, edge_type=None):
        if start_id == end_id:
            return [start_id]
        adj = self._load_adj(edge_type)
        queue: deque = deque([[start_id]])
        visited = {start_id}
        while queue:
            path = queue.popleft()
            for neighbor in dict.fromkeys(adj.get(path[-1], [])):
                if neighbor == end_id:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return []

    def dfs(self, start_id, end_id, edge_type=None):
        if start_id == end_id:
            return [start_id]
        adj = self._load_adj(edge_type)

        def _dfs(node, visited, path):
            for neighbor in dict.fromkeys(adj.get(node, [])):
                if neighbor == end_id:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    result = _dfs(neighbor, visited, path + [neighbor])
                    if result:
                        return result
            return []

        return _dfs(start_id, {start_id}, [start_id])

    def connected_components(self, edge_type=None):
        adj = self._load_adj(edge_type)
        with self._conn() as conn:
            all_nodes = [r[0] for r in conn.execute("SELECT node_id FROM nodes").fetchall()]
        visited: set = set()
        components = []
        for start in all_nodes:
            if start in visited:
                continue
            component = []
            queue: deque = deque([start])
            visited.add(start)
            while queue:
                node = queue.popleft()
                component.append(node)
                for neighbor in dict.fromkeys(adj.get(node, [])):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
        components.sort(key=len, reverse=True)
        return components

    def node_count(self):
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

    def edge_count(self, directed=False):
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        return total if directed else total // 2

    def edge_type_counts(self):
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT edge_type, COUNT(*) / 2 AS cnt FROM edges GROUP BY edge_type"
            ).fetchall()
        return {r[0]: r[1] for r in rows}
