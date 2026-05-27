"""Tests for the User-Movie Bipartite Graph (collab_graph module)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from databaseai.collab_graph import CollabGraph
from databaseai.seed_data import MOVIES, USERS, RATINGS


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def graph():
    g = CollabGraph()
    g.build_from_seed(MOVIES, USERS, RATINGS)
    return g


@pytest.fixture
def small_graph():
    """Minimal 3-user × 3-movie graph for deterministic assertions."""
    g = CollabGraph()
    g.add_user_node("u1", "alice")
    g.add_user_node("u2", "bob")
    g.add_user_node("u3", "carol")
    g.add_movie_node("m1", "Movie A")
    g.add_movie_node("m2", "Movie B")
    g.add_movie_node("m3", "Movie C")
    # u1 rated m1, m2; u2 rated m1, m3; u3 rated m2, m3
    g.add_rating_edge("u1", "m1", 5.0)
    g.add_rating_edge("u1", "m2", 4.0)
    g.add_rating_edge("u2", "m1", 4.5)
    g.add_rating_edge("u2", "m3", 3.5)
    g.add_rating_edge("u3", "m2", 4.0)
    g.add_rating_edge("u3", "m3", 5.0)
    return g


# ── Construction & counts ─────────────────────────────────────────────────────

class TestConstruction:

    def test_node_counts_match_seed(self, graph):
        counts = graph.node_counts()
        assert counts["users"]  == len(USERS)
        assert counts["movies"] == len(MOVIES)
        assert counts["edges"]  == len(RATINGS)

    def test_add_duplicate_node_is_idempotent(self):
        g = CollabGraph()
        g.add_user_node("u1", "alice")
        g.add_user_node("u1", "alice_again")  # second insert should be ignored
        assert g.node_counts()["users"] == 1

    def test_add_rating_edge_upserts_weight(self):
        g = CollabGraph()
        g.add_user_node("u1", "alice")
        g.add_movie_node("m1", "Movie A")
        g.add_rating_edge("u1", "m1", 3.0)
        g.add_rating_edge("u1", "m1", 5.0)  # update
        movies = g.user_rated_movies("u1")
        assert len(movies) == 1
        assert movies[0]["rating"] == 5.0

    def test_small_graph_edge_count(self, small_graph):
        assert small_graph.node_counts()["edges"] == 6


# ── Traversal ─────────────────────────────────────────────────────────────────

class TestTraversal:

    def test_user_rated_movies_returns_all_rated(self, graph):
        expected_count = sum(1 for r in RATINGS if r[0] == "u01")
        assert len(graph.user_rated_movies("u01")) == expected_count

    def test_user_rated_movies_sorted_descending(self, graph):
        movies = graph.user_rated_movies("u01")
        ratings = [m["rating"] for m in movies]
        assert ratings == sorted(ratings, reverse=True)

    def test_movie_raters_contains_expected_users(self, graph):
        # m01 is rated by u01, u02, u03, u04 in seed data
        rater_ids = {r["user_id"] for r in graph.movie_raters("m01")}
        assert {"u01", "u02", "u03", "u04"}.issubset(rater_ids)

    def test_movie_raters_sorted_descending(self, graph):
        raters = graph.movie_raters("m01")
        ratings = [r["rating"] for r in raters]
        assert ratings == sorted(ratings, reverse=True)

    def test_unrated_movie_has_no_raters(self):
        g = CollabGraph()
        g.add_movie_node("mx", "Mystery")
        assert g.movie_raters("mx") == []

    def test_new_user_has_no_rated_movies(self):
        g = CollabGraph()
        g.add_user_node("unew", "stranger")
        assert g.user_rated_movies("unew") == []


# ── Collaborative recommendations ─────────────────────────────────────────────

class TestCollabRecommendations:

    def test_recs_exclude_already_rated(self, graph):
        rated = {r["movie_id"] for r in graph.user_rated_movies("u01")}
        recs = graph.collab_recommendations("u01")
        rec_ids = {r["movie_id"] for r in recs}
        assert len(rec_ids & rated) == 0, "Must not recommend already-rated movies"

    def test_recs_have_positive_votes(self, graph):
        recs = graph.collab_recommendations("u01", top_n=5)
        assert all(r["co_rater_votes"] >= 1 for r in recs)

    def test_recs_sorted_by_votes_desc(self, graph):
        recs = graph.collab_recommendations("u01", top_n=5)
        votes = [r["co_rater_votes"] for r in recs]
        assert votes == sorted(votes, reverse=True)

    def test_cold_start_returns_empty(self):
        g = CollabGraph()
        g.add_user_node("unew", "stranger")
        assert g.collab_recommendations("unew") == []

    def test_recs_top_n_respects_limit(self, graph):
        for limit in [1, 3, 5]:
            recs = graph.collab_recommendations("u01", top_n=limit)
            assert len(recs) <= limit

    def test_2hop_produces_recommendations_in_small_graph(self, small_graph):
        # u1 rated m1,m2; u2 (co-rater via m1) also rated m3 → m3 should appear
        recs = small_graph.collab_recommendations("u1")
        rec_ids = {r["movie_id"] for r in recs}
        assert "m3" in rec_ids


# ── Similarity ────────────────────────────────────────────────────────────────

class TestSimilarity:

    def test_jaccard_identical_users(self, small_graph):
        assert small_graph.user_similarity("u1", "u1") == 1.0

    def test_jaccard_disjoint_users(self):
        g = CollabGraph()
        g.add_user_node("ua", "Alice")
        g.add_user_node("ub", "Bob")
        g.add_movie_node("m1", "Movie A")
        g.add_movie_node("m2", "Movie B")
        g.add_rating_edge("ua", "m1", 5.0)
        g.add_rating_edge("ub", "m2", 5.0)
        assert g.user_similarity("ua", "ub") == 0.0

    def test_jaccard_partial_overlap(self, small_graph):
        # u1 rated {m1,m2}; u2 rated {m1,m3} → overlap={m1}, union={m1,m2,m3} → 1/3
        sim = small_graph.user_similarity("u1", "u2")
        assert abs(sim - 1 / 3) < 1e-9

    def test_jaccard_full_overlap_subset(self):
        g = CollabGraph()
        g.add_user_node("ua", "Alice")
        g.add_user_node("ub", "Bob")
        g.add_movie_node("m1", "Movie A")
        g.add_rating_edge("ua", "m1", 5.0)
        g.add_rating_edge("ub", "m1", 3.0)
        assert g.user_similarity("ua", "ub") == 1.0

    def test_most_similar_sorted_desc(self, graph):
        similar = graph.most_similar_users("u01", top_n=4)
        sims = [s["similarity"] for s in similar]
        assert sims == sorted(sims, reverse=True)

    def test_most_similar_excludes_self(self, graph):
        similar = graph.most_similar_users("u01")
        assert all(s["user_id"] != "u01" for s in similar)

    def test_similarity_matrix_all_pairs(self, small_graph):
        pairs = small_graph.similarity_matrix()
        # 3 users → 3 choose 2 = 3 pairs
        assert len(pairs) == 3
        assert all("user_a" in p and "user_b" in p and "similarity" in p for p in pairs)


# ── Popularity ────────────────────────────────────────────────────────────────

class TestPopularity:

    def test_movie_popularity_sorted_by_rater_count(self, graph):
        pop = graph.movie_popularity(top_n=5)
        counts = [m["rater_count"] for m in pop]
        assert counts == sorted(counts, reverse=True)

    def test_most_rated_has_four_raters(self, graph):
        # m01 and m07 each have 4 raters in seed data
        pop = graph.movie_popularity(top_n=1)
        assert pop[0]["rater_count"] == 4

    def test_popularity_top_n_limit(self, graph):
        for limit in [1, 5, 10]:
            pop = graph.movie_popularity(top_n=limit)
            assert len(pop) <= limit


# ── Similarity-weighted recommendations ──────────────────────────────────────

class TestWeightedRecommendations:

    def test_weighted_recs_exclude_already_rated(self, graph):
        rated = {r["movie_id"] for r in graph.user_rated_movies("u01")}
        recs = graph.similarity_weighted_recommendations("u01")
        rec_ids = {r["movie_id"] for r in recs}
        assert len(rec_ids & rated) == 0

    def test_weighted_recs_have_positive_scores(self, graph):
        recs = graph.similarity_weighted_recommendations("u01", top_n=5)
        assert all(r["weighted_score"] > 0 for r in recs)

    def test_weighted_recs_sorted_desc(self, graph):
        recs = graph.similarity_weighted_recommendations("u01", top_n=5)
        scores = [r["weighted_score"] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_cold_start_weighted_returns_empty(self):
        g = CollabGraph()
        g.add_user_node("unew", "stranger")
        assert g.similarity_weighted_recommendations("unew") == []
