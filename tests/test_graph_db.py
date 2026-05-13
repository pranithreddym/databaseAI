"""Tests for the Movie Graph (Graph Database module)."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from databaseai.seed_data import MOVIES
from databaseai.graph_db import MovieGraph, CAST


@pytest.fixture
def graph():
    g = MovieGraph()
    g.build_from_seed(MOVIES, cast=CAST)
    return g


@pytest.fixture
def small_graph():
    g = MovieGraph()
    for nid, label in [("A", "Alpha"), ("B", "Beta"), ("C", "Gamma"), ("D", "Delta")]:
        g.add_node(nid, label)
    g.add_edge("A", "B", "link")
    g.add_edge("B", "C", "link")
    return g


class TestGraphConstruction:

    def test_node_count_equals_movie_count(self, graph):
        assert graph.node_count() == len(MOVIES)

    def test_edge_types_present(self, graph):
        etypes = set(graph.edge_type_counts().keys())
        assert "same_genre" in etypes and "same_director" in etypes and "co_actor" in etypes

    def test_same_director_edges_nolan(self, graph):
        neighbors_m01_dir = {nb["node_id"] for nb in graph.neighbors("m01", edge_type="same_director")}
        assert "m02" in neighbors_m01_dir and "m03" in neighbors_m01_dir

    def test_same_genre_edges_scifi(self, graph):
        scifi_ids = {m["id"] for m in MOVIES if m.get("genre") == "sci-fi"} - {"m01"}
        neighbors = {nb["node_id"] for nb in graph.neighbors("m01", edge_type="same_genre")}
        assert scifi_ids == neighbors

    def test_co_actor_edge_michael_caine(self, graph):
        nb_m01_actor = {nb["node_id"] for nb in graph.neighbors("m01", edge_type="co_actor")}
        assert "m02" in nb_m01_actor and "m03" in nb_m01_actor

    def test_edges_are_bidirectional(self, graph):
        fwd = {nb["node_id"] for nb in graph.neighbors("m01", edge_type="same_director")}
        rev = {nb["node_id"] for nb in graph.neighbors("m02", edge_type="same_director")}
        assert "m02" in fwd and "m01" in rev

    def test_undirected_edge_count_is_half_directed(self, graph):
        assert graph.edge_count(directed=True) == graph.edge_count(directed=False) * 2

    def test_duplicate_build_does_not_double_edges(self, graph):
        count_before = graph.edge_count()
        graph.build_from_seed(MOVIES, cast=CAST)
        assert graph.edge_count() == count_before


class TestNeighbors:

    def test_neighbors_returns_list_of_dicts(self, graph):
        nbs = graph.neighbors("m01")
        assert isinstance(nbs, list) and all(isinstance(n, dict) for n in nbs)

    def test_neighbors_have_required_keys(self, graph):
        for nb in graph.neighbors("m01"):
            assert "node_id" in nb and "edge_type" in nb and "label" in nb

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


class TestBFS:

    def test_bfs_same_node_returns_singleton(self, graph):
        assert graph.bfs("m01", "m01") == ["m01"]

    def test_bfs_direct_connection_two_nodes(self, graph):
        path = graph.bfs("m01", "m02", edge_type="same_director")
        assert path == ["m01", "m02"]

    def test_bfs_returns_shortest_path(self, small_graph):
        assert small_graph.bfs("A", "C") == ["A", "B", "C"]

    def test_bfs_no_path_disconnected_returns_empty(self, small_graph):
        assert small_graph.bfs("A", "D") == []

    def test_bfs_cross_genre_via_coactor(self, graph):
        path = graph.bfs("m14", "m18", edge_type="co_actor")
        assert path == ["m14", "m18"]

    def test_bfs_cross_genre_multi_hop(self, graph):
        path = graph.bfs("m05", "m01")
        assert len(path) >= 2 and path[0] == "m05" and path[-1] == "m01"

    def test_bfs_path_is_valid_in_graph(self, graph):
        path = graph.bfs("m05", "m01")
        assert path
        for i in range(len(path) - 1):
            neighbor_ids = {nb["node_id"] for nb in graph.neighbors(path[i])}
            assert path[i + 1] in neighbor_ids

    def test_bfs_drama_to_animation_no_path(self, graph):
        assert graph.bfs("m19", "m08") == []


class TestDFS:

    def test_dfs_same_node_returns_singleton(self, graph):
        assert graph.dfs("m01", "m01") == ["m01"]

    def test_dfs_finds_path_between_connected_nodes(self, small_graph):
        path = small_graph.dfs("A", "C")
        assert len(path) >= 2 and path[0] == "A" and path[-1] == "C"

    def test_dfs_no_path_returns_empty(self, small_graph):
        assert small_graph.dfs("A", "D") == []

    def test_dfs_path_is_valid_in_graph(self, graph):
        path = graph.dfs("m05", "m01")
        assert path
        for i in range(len(path) - 1):
            neighbor_ids = {nb["node_id"] for nb in graph.neighbors(path[i])}
            assert path[i + 1] in neighbor_ids

    def test_bfs_hop_count_lte_dfs(self, graph):
        bfs_path = graph.bfs("m05", "m01")
        dfs_path = graph.dfs("m05", "m01")
        if bfs_path and dfs_path:
            assert len(bfs_path) <= len(dfs_path)


class TestConnectedComponents:

    def test_components_cover_all_nodes(self, graph):
        comps = graph.connected_components()
        all_in_comps = {nid for c in comps for nid in c}
        assert all_in_comps == {m["id"] for m in MOVIES}

    def test_multiple_components_exist(self, graph):
        assert len(graph.connected_components()) >= 2

    def test_largest_component_contains_scifi_and_action(self, graph):
        largest = set(graph.connected_components()[0])
        assert "m01" in largest and "m02" in largest

    def test_drama_cluster_is_isolated(self, graph):
        comps = graph.connected_components()
        largest = set(comps[0])
        drama_ids = {"m07", "m15", "m19"}
        assert not drama_ids.intersection(largest)

    def test_isolated_node_forms_own_component(self):
        g = MovieGraph()
        g.add_node("X", "Standalone")
        g.add_node("Y", "Also Standalone")
        g.add_edge("X", "Y", "link")
        g.add_node("Z", "Isolated")
        comp_sets = [set(c) for c in g.connected_components()]
        assert {"Z"} in comp_sets and {"X", "Y"} in comp_sets


class TestDegreeCentrality:

    def test_most_connected_returns_n_results(self, graph):
        assert len(graph.most_connected(n=5)) == 5

    def test_most_connected_result_has_required_keys(self, graph):
        for row in graph.most_connected(n=3):
            assert "node_id" in row and "label" in row and "degree" in row

    def test_most_connected_ordered_descending(self, graph):
        top = graph.most_connected(n=10)
        degrees = [row["degree"] for row in top]
        assert degrees == sorted(degrees, reverse=True)

    def test_sci_fi_movies_have_high_degree(self, graph):
        assert graph.degree("m01") >= 6

    def test_degree_filtered_by_edge_type(self, graph):
        assert graph.degree("m01", edge_type="same_director") == 2


class TestGetNode:

    def test_get_node_returns_dict(self, graph):
        node = graph.get_node("m01")
        assert isinstance(node, dict) and node["node_id"] == "m01" and "label" in node

    def test_get_node_missing_returns_none(self, graph):
        assert graph.get_node("nonexistent") is None
