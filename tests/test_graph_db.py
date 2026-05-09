"""Tests for the Movie Graph (Graph Database module)."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from databaseai.seed_data import MOVIES
from databaseai.graph_db import MovieGraph, CAST


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def graph():
    """Full movie graph built from seed data with cast."""
    g = MovieGraph()
    g.build_from_seed(MOVIES, cast=CAST)
    return g


@pytest.fixture
def small_graph():
    """Minimal 4-node graph for isolated algorithm tests."""
    g = MovieGraph()
    for nid, label in [("A", "Alpha"), ("B", "Beta"), ("C", "Gamma"), ("D", "Delta")]:
        g.add_node(nid, label)
    # A--B--C  D is isolated
    g.add_edge("A", "B", "link")
    g.add_edge("B", "C", "link")
    return g


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

class TestGraphConstruction:

    def test_node_count_equals_movie_count(self, graph):
        assert graph.node_count() == len(MOVIES)

    def test_edge_types_present(self, graph):
        etypes = set(graph.edge_type_counts().keys())
        assert "same_genre" in etypes
        assert "same_director" in etypes
        assert "co_actor" in etypes

    def test_same_director_edges_nolan(self, graph):
        # Inception (m01), Dark Knight (m02), Interstellar (m03) share Nolan
        neighbors_m01_dir = {
            nb["node_id"] for nb in graph.neighbors("m01", edge_type="same_director")
        }
        assert "m02" in neighbors_m01_dir
        assert "m03" in neighbors_m01_dir

    def test_same_genre_edges_scifi(self, graph):
        # All sci-fi movies should be neighbors of Inception (m01)
        scifi_ids = {m["id"] for m in MOVIES if m.get("genre") == "sci-fi"} - {"m01"}
        neighbors_m01_genre = {
            nb["node_id"] for nb in graph.neighbors("m01", edge_type="same_genre")
        }
        assert scifi_ids == neighbors_m01_genre

    def test_co_actor_edge_michael_caine(self, graph):
        # Michael Caine appears in m01, m02, m03 → co_actor edges
        nb_m01_actor = {
            nb["node_id"] for nb in graph.neighbors("m01", edge_type="co_actor")
        }
        assert "m02" in nb_m01_actor
        assert "m03" in nb_m01_actor

    def test_edges_are_bidirectional(self, graph):
        # If m01 lists m02 as a same_director neighbor, m02 must list m01 too
        fwd = {nb["node_id"] for nb in graph.neighbors("m01", edge_type="same_director")}
        rev = {nb["node_id"] for nb in graph.neighbors("m02", edge_type="same_director")}
        assert "m02" in fwd
        assert "m01" in rev

    def test_undirected_edge_count_is_half_directed(self, graph):
        directed = graph.edge_count(directed=True)
        undirected = graph.edge_count(directed=False)
        assert directed == undirected * 2

    def test_duplicate_build_does_not_double_edges(self, graph):
        count_before = graph.edge_count()
        graph.build_from_seed(MOVIES, cast=CAST)   # second call — INSERT OR IGNORE
        assert graph.edge_count() == count_before


# ---------------------------------------------------------------------------
# Neighbor queries
# ---------------------------------------------------------------------------

class TestNeighbors:

    def test_neighbors_returns_list_of_dicts(self, graph):
        nbs = graph.neighbors("m01")
        assert isinstance(nbs, list)
        assert all(isinstance(n, dict) for n in nbs)

    def test_neighbors_have_required_keys(self, graph):
        nbs = graph.neighbors("m01")
        for nb in nbs:
            assert "node_id" in nb
            assert "edge_type" in nb
            assert "label" in nb

    def test_neighbors_filtered_by_edge_type(self, graph):
        genre_nbs = graph.neighbors("m01", edge_type="same_genre")
        assert all(nb["edge_type"] == "same_genre" for nb in genre_nbs)

    def test_isolated_node_has_no_neighbors(self):
        g = MovieGraph()
        g.add_node("lone", "Lone Movie")
        assert g.neighbors("lone") == []

    def test_degree_matches_unique_neighbor_count(self, graph):
        nbs = graph.neighbors("m01", edge_type="same_director")
        unique_ids = {nb["node_id"] for nb in nbs}
        assert graph.degree("m01", edge_type="same_director") == len(unique_ids)


# ---------------------------------------------------------------------------
# BFS
# ---------------------------------------------------------------------------

class TestBFS:

    def test_bfs_same_node_returns_singleton(self, graph):
        assert graph.bfs("m01", "m01") == ["m01"]

    def test_bfs_direct_connection_two_nodes(self, graph):
        # m01 and m02 share director — path length = 1 hop
        path = graph.bfs("m01", "m02", edge_type="same_director")
        assert path == ["m01", "m02"]

    def test_bfs_returns_shortest_path(self, small_graph):
        # A-B-C: BFS A→C should return exactly 2 hops
        path = small_graph.bfs("A", "C")
        assert path == ["A", "B", "C"]

    def test_bfs_no_path_disconnected_returns_empty(self, small_graph):
        # D is isolated
        assert small_graph.bfs("A", "D") == []

    def test_bfs_cross_genre_via_coactor(self, graph):
        # Knives Out (m14, thriller) → Blade Runner (m18, sci-fi) via Ana de Armas
        path = graph.bfs("m14", "m18", edge_type="co_actor")
        assert path == ["m14", "m18"]

    def test_bfs_cross_genre_multi_hop(self, graph):
        # Parasite (m05, thriller) → Inception (m01, sci-fi) within a few hops
        path = graph.bfs("m05", "m01")
        assert len(path) >= 2                   # at least 1 hop
        assert path[0] == "m05"
        assert path[-1] == "m01"

    def test_bfs_path_is_valid_in_graph(self, graph):
        """Every consecutive pair in the BFS result must be actual neighbors."""
        path = graph.bfs("m05", "m01")
        assert path, "Expected a valid path from m05 to m01"
        for i in range(len(path) - 1):
            neighbor_ids = {nb["node_id"] for nb in graph.neighbors(path[i])}
            assert path[i + 1] in neighbor_ids, (
                f"{path[i + 1]} is not a neighbor of {path[i]}"
            )

    def test_bfs_drama_to_animation_no_path(self, graph):
        # Drama cluster (m19) and animation cluster (m08) have no bridging edges
        path = graph.bfs("m19", "m08")
        assert path == []


# ---------------------------------------------------------------------------
# DFS
# ---------------------------------------------------------------------------

class TestDFS:

    def test_dfs_same_node_returns_singleton(self, graph):
        assert graph.dfs("m01", "m01") == ["m01"]

    def test_dfs_finds_path_between_connected_nodes(self, small_graph):
        path = small_graph.dfs("A", "C")
        assert len(path) >= 2
        assert path[0] == "A"
        assert path[-1] == "C"

    def test_dfs_no_path_returns_empty(self, small_graph):
        assert small_graph.dfs("A", "D") == []

    def test_dfs_path_is_valid_in_graph(self, graph):
        """Every consecutive pair in the DFS result must be actual neighbors."""
        path = graph.dfs("m05", "m01")
        assert path, "Expected a valid path from m05 to m01"
        for i in range(len(path) - 1):
            neighbor_ids = {nb["node_id"] for nb in graph.neighbors(path[i])}
            assert path[i + 1] in neighbor_ids, (
                f"{path[i + 1]} is not a neighbor of {path[i]}"
            )

    def test_bfs_hop_count_lte_dfs(self, graph):
        # BFS must never return a longer path than DFS for the same query
        bfs_path = graph.bfs("m05", "m01")
        dfs_path = graph.dfs("m05", "m01")
        if bfs_path and dfs_path:
            assert len(bfs_path) <= len(dfs_path)


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------

class TestConnectedComponents:

    def test_components_cover_all_nodes(self, graph):
        comps = graph.connected_components()
        all_in_comps = {nid for c in comps for nid in c}
        all_nodes = {m["id"] for m in MOVIES}
        assert all_in_comps == all_nodes

    def test_multiple_components_exist(self, graph):
        # Drama, animation, horror clusters are isolated from the main supercluster
        comps = graph.connected_components()
        assert len(comps) >= 2

    def test_largest_component_contains_scifi_and_action(self, graph):
        largest = set(graph.connected_components()[0])
        # sci-fi hub (m01) and action hub (m02) should be in the same component
        assert "m01" in largest
        assert "m02" in largest

    def test_drama_cluster_is_isolated(self, graph):
        # m07, m15, m19 share only same_genre=drama and have no cast bridge
        # Verify they're NOT in the main sci-fi/action/thriller supercluster
        comps = graph.connected_components()
        largest = set(comps[0])
        drama_ids = {"m07", "m15", "m19"}
        assert not drama_ids.intersection(largest), (
            "Drama movies should be in an isolated component"
        )

    def test_isolated_node_forms_own_component(self):
        g = MovieGraph()
        g.add_node("X", "Standalone")
        g.add_node("Y", "Also Standalone")
        g.add_edge("X", "Y", "link")
        g.add_node("Z", "Isolated")
        comps = g.connected_components()
        comp_sets = [set(c) for c in comps]
        assert {"Z"} in comp_sets
        assert {"X", "Y"} in comp_sets


# ---------------------------------------------------------------------------
# Degree centrality
# ---------------------------------------------------------------------------

class TestDegreeCentrality:

    def test_most_connected_returns_n_results(self, graph):
        top5 = graph.most_connected(n=5)
        assert len(top5) == 5

    def test_most_connected_result_has_required_keys(self, graph):
        for row in graph.most_connected(n=3):
            assert "node_id" in row
            assert "label" in row
            assert "degree" in row

    def test_most_connected_ordered_descending(self, graph):
        top = graph.most_connected(n=10)
        degrees = [row["degree"] for row in top]
        assert degrees == sorted(degrees, reverse=True)

    def test_sci_fi_movies_have_high_degree(self, graph):
        # Inception (m01) connects to 6 other sci-fi films + 2 Nolan films + co-actors
        assert graph.degree("m01") >= 6

    def test_degree_filtered_by_edge_type(self, graph):
        # Inception (m01) should have exactly 2 same_director neighbors
        # (Dark Knight m02 and Interstellar m03)
        assert graph.degree("m01", edge_type="same_director") == 2


# ---------------------------------------------------------------------------
# get_node
# ---------------------------------------------------------------------------

class TestGetNode:

    def test_get_node_returns_dict(self, graph):
        node = graph.get_node("m01")
        assert isinstance(node, dict)
        assert node["node_id"] == "m01"
        assert "label" in node
        assert "genre" in node

    def test_get_node_missing_returns_none(self, graph):
        assert graph.get_node("nonexistent") is None
