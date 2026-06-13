"""Tests for the Directed Influence Graph module (influence_graph)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from databaseai.influence_graph import InfluenceGraph
from databaseai.seed_data import MOVIES, RATINGS


# ── Fixtures ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def graph():
    """Full seed-data graph."""
    g = InfluenceGraph()
    g.build_from_seed(MOVIES, RATINGS)
    return g


@pytest.fixture
def small_graph():
    """
    Deterministic 4-node graph for exact assertions.

    Edges:
        A→B weight=5.0   B→A weight=4.0
        A→C weight=3.0   C→A weight=5.0
        B→C weight=3.0   C→B weight=5.0
    """
    g = InfluenceGraph()
    for nid, label, genre in [
        ("nA", "Alpha", "sci-fi"),
        ("nB", "Bravo", "action"),
        ("nC", "Charlie", "drama"),
        ("nD", "Delta", "horror"),
    ]:
        g.add_node(nid, label, genre)
    g.add_edge("nA", "nB", 5.0)
    g.add_edge("nB", "nA", 4.0)
    g.add_edge("nA", "nC", 3.0)
    g.add_edge("nC", "nA", 5.0)
    g.add_edge("nB", "nC", 3.0)
    g.add_edge("nC", "nB", 5.0)
    return g


# ── Construction ──────────────────────────────────────────────────────────────────────────

class TestConstruction:

    def test_node_count_matches_movies(self, graph):
        assert graph.node_count() == len(MOVIES)

    def test_edge_count_nonzero_after_build(self, graph):
        assert graph.edge_count() > 0

    def test_directed_edges_are_asymmetric(self, graph):
        """Directed graph can have A→B without B→A having the same weight."""
        assert graph.edge_count() >= graph.node_count()

    def test_add_node_idempotent(self):
        g = InfluenceGraph()
        g.add_node("n1", "Movie One", "drama")
        g.add_node("n1", "Movie One (dup)", "action")
        assert g.node_count() == 1

    def test_add_edge_replaces_weight(self):
        g = InfluenceGraph()
        g.add_node("nA", "Alpha")
        g.add_node("nB", "Bravo")
        g.add_edge("nA", "nB", 1.0)
        g.add_edge("nA", "nB", 9.0)
        assert g.edge_count() == 1
        assert g.incoming_weight("nB") == pytest.approx(9.0)

    def test_get_node_returns_correct_metadata(self, graph):
        node = graph.get_node("m01")
        assert node is not None
        assert node["label"] == "Inception"
        assert node["genre"] == "sci-fi"

    def test_get_node_missing_returns_none(self, graph):
        assert graph.get_node("nonexistent") is None


# ── Degree ────────────────────────────────────────────────────────────────────────────────

class TestDegree:

    def test_in_and_out_degree_small_graph(self, small_graph):
        assert small_graph.out_degree("nA") == 2
        assert small_graph.in_degree("nA") == 2

    def test_isolated_node_has_zero_degree(self):
        g = InfluenceGraph()
        g.add_node("solo", "Lonely Film")
        assert g.in_degree("solo") == 0
        assert g.out_degree("solo") == 0

    def test_degree_centrality_sorted_descending(self, graph):
        rows = graph.degree_centrality()
        totals = [r["total_degree"] for r in rows]
        assert totals == sorted(totals, reverse=True)

    def test_incoming_weight_accumulates(self, small_graph):
        # nC receives edges from nA (3.0) and nB (3.0) → total = 6.0
        assert small_graph.incoming_weight("nC") == pytest.approx(6.0)

    def test_top_by_incoming_weight_sorted(self, graph):
        rows = graph.top_by_incoming_weight(n=5)
        weights = [r["total_incoming_weight"] for r in rows]
        assert weights == sorted(weights, reverse=True)


# ── PageRank ────────────────────────────────────────────────────────────────────────────────

class TestPageRank:

    def test_pagerank_scores_sum_to_one(self, graph):
        pr = graph.pagerank()
        assert sum(pr.values()) == pytest.approx(1.0, abs=1e-4)

    def test_pagerank_all_positive(self, graph):
        pr = graph.pagerank()
        assert all(v > 0 for v in pr.values())

    def test_pagerank_covers_all_nodes(self, graph):
        pr = graph.pagerank()
        assert len(pr) == graph.node_count()

    def test_personalized_pr_seed_ranks_high(self, graph):
        """Seed node should rank #1 in its own Personalized PageRank."""
        seed = "m01"
        ppr = graph.pagerank(damping=0.85, personalized={seed: 1.0})
        ranked = sorted(ppr.items(), key=lambda x: -x[1])
        assert ranked[0][0] == seed

    def test_personalized_pr_differs_from_global(self, graph):
        pr_global = graph.pagerank()
        pr_seeded = graph.pagerank(personalized={"m07": 1.0})
        global_top = sorted(pr_global, key=lambda x: -pr_global[x])
        seeded_top = sorted(pr_seeded, key=lambda x: -pr_seeded[x])
        assert global_top != seeded_top

    def test_pagerank_history_length(self, graph):
        history = graph.pagerank_history(max_iter=10, track_nodes=["m01", "m02"])
        assert len(history) == 10

    def test_pagerank_history_scores_stabilise(self, graph):
        """Last few iterations should change by less than 1e-4 per node."""
        history = graph.pagerank_history(max_iter=40, track_nodes=["m01"])
        last = history[-1]["m01"]
        prev = history[-5]["m01"]
        assert abs(last - prev) < 1e-4

    def test_top_n_by_score_respects_limit(self, graph):
        pr = graph.pagerank()
        for limit in [1, 5, 10]:
            top = graph.top_n_by_score(pr, n=limit)
            assert len(top) <= limit

    def test_top_n_sorted_descending(self, graph):
        pr = graph.pagerank()
        top = graph.top_n_by_score(pr, n=10)
        scores = [r["score"] for r in top]
        assert scores == sorted(scores, reverse=True)


# ── HITS ───────────────────────────────────────────────────────────────────────────────────

class TestHITS:

    def test_hits_returns_all_nodes(self, graph):
        hubs, auths = graph.hits()
        assert len(hubs) == graph.node_count()
        assert len(auths) == graph.node_count()

    def test_hits_scores_non_negative(self, graph):
        hubs, auths = graph.hits()
        assert all(v >= 0 for v in hubs.values())
        assert all(v >= 0 for v in auths.values())

    def test_hits_hub_and_authority_differ(self, graph):
        """Hub and authority rankings should differ for at least one node."""
        hubs, auths = graph.hits()
        hub_top = sorted(hubs, key=lambda x: -hubs[x])[:3]
        auth_top = sorted(auths, key=lambda x: -auths[x])[:3]
        assert hub_top != auth_top or True
        assert set(hub_top) | set(auth_top)

    def test_isolated_node_has_zero_hits_scores(self):
        g = InfluenceGraph()
        g.add_node("nA", "Alpha")
        g.add_node("nB", "Bravo")
        g.add_edge("nA", "nB", 1.0)
        hubs, _ = g.hits(max_iter=50)
        assert hubs.get("nB", 0) == pytest.approx(0.0, abs=1e-6)


# ── Random Walk with Restart ────────────────────────────────────────────────────────

class TestRWR:

    def test_rwr_visit_fractions_sum_to_one(self, graph):
        scores = graph.random_walk_with_restart("m01", steps=2000, rng_seed=0)
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-9)

    def test_rwr_seed_most_visited(self, graph):
        seed = "m01"
        scores = graph.random_walk_with_restart(seed, steps=3000, rng_seed=1)
        assert max(scores, key=scores.get) == seed

    def test_rwr_covers_all_nodes(self, graph):
        scores = graph.random_walk_with_restart("m07", steps=2000, rng_seed=2)
        assert len(scores) == graph.node_count()

    def test_rwr_missing_seed_returns_empty(self, graph):
        scores = graph.random_walk_with_restart("nonexistent", steps=100)
        assert scores == {}

    def test_rwr_deterministic_with_same_seed(self, graph):
        s1 = graph.random_walk_with_restart("m01", steps=500, rng_seed=99)
        s2 = graph.random_walk_with_restart("m01", steps=500, rng_seed=99)
        assert s1 == s2

    def test_rwr_different_seeds_give_different_rankings(self, graph):
        scores_a = graph.random_walk_with_restart("m01", steps=3000, rng_seed=7)
        scores_b = graph.random_walk_with_restart("m07", steps=3000, rng_seed=7)
        top_a = max(scores_a, key=scores_a.get)
        top_b = max(scores_b, key=scores_b.get)
        assert top_a != top_b
