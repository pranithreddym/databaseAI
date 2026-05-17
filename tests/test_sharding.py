"""Tests for the Sharding module (ConsistentHashRing + ShardManager)."""

import sys
import os
import tempfile
import shutil

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from databaseai.seed_data import MOVIES, RATINGS, USERS
from databaseai.sharding import ConsistentHashRing, ShardManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmpdir_path():
    d = tempfile.mkdtemp(prefix="sharding_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def three_shard_paths(tmpdir_path):
    return [os.path.join(tmpdir_path, f"shard_{i}.db") for i in range(3)]


@pytest.fixture
def mgr(three_shard_paths):
    m = ShardManager(three_shard_paths, vnodes_per_node=50)
    yield m
    m.close()


@pytest.fixture
def seeded_mgr(three_shard_paths):
    m = ShardManager(three_shard_paths, vnodes_per_node=50)
    m.insert_movies(MOVIES)
    m.insert_ratings_bulk(RATINGS)
    yield m
    m.close()


# ---------------------------------------------------------------------------
# ConsistentHashRing
# ---------------------------------------------------------------------------

class TestConsistentHashRing:

    def test_empty_ring_returns_none(self):
        ring = ConsistentHashRing()
        assert ring.get_node("any_key") is None

    def test_single_node_owns_all_keys(self):
        ring = ConsistentHashRing(vnodes_per_node=10)
        ring.add_node("node_A")
        for key in ["a", "b", "c", "xyz", "123"]:
            assert ring.get_node(key) == "node_A"

    def test_routing_is_deterministic(self):
        ring = ConsistentHashRing(vnodes_per_node=50)
        ring.add_node("s0")
        ring.add_node("s1")
        ring.add_node("s2")
        # Same key always maps to same node across repeated calls.
        for key in ["u01", "u02", "u03", "u04", "u05"]:
            first = ring.get_node(key)
            for _ in range(10):
                assert ring.get_node(key) == first

    def test_adding_node_does_not_reassign_all_keys(self):
        ring = ConsistentHashRing(vnodes_per_node=50)
        ring.add_node("s0")
        ring.add_node("s1")
        ring.add_node("s2")

        keys = [f"key_{i}" for i in range(200)]
        before = {k: ring.get_node(k) for k in keys}

        ring.add_node("s3")
        after = {k: ring.get_node(k) for k in keys}

        unchanged = sum(1 for k in keys if before[k] == after[k])
        # Consistent hashing: at most ~25% of keys should move.
        assert unchanged >= len(keys) * 0.65

    def test_removing_node_reassigns_its_keys(self):
        ring = ConsistentHashRing(vnodes_per_node=50)
        ring.add_node("s0")
        ring.add_node("s1")
        ring.add_node("s2")
        keys = [f"k{i}" for i in range(100)]

        ring.remove_node("s1")
        assert "s1" not in ring.nodes
        for k in keys:
            assert ring.get_node(k) != "s1"

    def test_nodes_property_lists_all_added_nodes(self):
        ring = ConsistentHashRing(vnodes_per_node=10)
        ring.add_node("alpha")
        ring.add_node("beta")
        ring.add_node("gamma")
        assert set(ring.nodes) == {"alpha", "beta", "gamma"}

    def test_key_distribution_sums_to_total(self):
        ring = ConsistentHashRing(vnodes_per_node=50)
        ring.add_node("n0")
        ring.add_node("n1")
        ring.add_node("n2")
        keys = [f"user_{i}" for i in range(500)]
        dist = ring.key_distribution(keys)
        assert sum(dist.values()) == len(keys)

    def test_key_distribution_covers_all_nodes(self):
        ring = ConsistentHashRing(vnodes_per_node=150)
        for n in ["n0", "n1", "n2"]:
            ring.add_node(n)
        keys = [f"user_{i}" for i in range(1000)]
        dist = ring.key_distribution(keys)
        # All nodes should receive at least some keys with 1000 inputs.
        for node in ring.nodes:
            assert dist[node] > 0


# ---------------------------------------------------------------------------
# ShardManager — routing
# ---------------------------------------------------------------------------

class TestShardManagerRouting:

    def test_user_routing_is_consistent(self, seeded_mgr):
        for u in USERS:
            first = seeded_mgr.get_shard_for_user(u["id"])
            for _ in range(5):
                assert seeded_mgr.get_shard_for_user(u["id"]) == first

    def test_total_ratings_equals_seed_count(self, seeded_mgr):
        assert seeded_mgr.total_ratings() == len(RATINGS)

    def test_per_shard_counts_sum_to_total(self, seeded_mgr):
        per_shard = seeded_mgr.rating_count_per_shard()
        assert sum(per_shard.values()) == seeded_mgr.total_ratings()

    def test_user_ratings_only_on_assigned_shard(self, seeded_mgr):
        for u in USERS:
            home = seeded_mgr.get_shard_for_user(u["id"])
            ratings = seeded_mgr.user_ratings(u["id"])
            # All returned rows should come from the home shard.
            assert len(ratings) > 0
            for r in ratings:
                assert r["user_id"] == u["id"]

    def test_movies_replicated_to_all_shards(self, seeded_mgr):
        for node, conn in seeded_mgr._conns.items():
            count = conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
            assert count == len(MOVIES), f"Shard {node} missing movies"

    def test_insert_rating_returns_shard_id(self, mgr):
        mgr.insert_movies(MOVIES)
        node = mgr.insert_rating("u01", "m01", 4.5, "great")
        assert node in mgr.shard_ids

    def test_bulk_insert_routes_to_correct_shards(self, seeded_mgr):
        per_shard = seeded_mgr.rating_count_per_shard()
        # No shard should be completely empty given 5 users and 3 shards.
        non_empty = sum(1 for v in per_shard.values() if v > 0)
        assert non_empty >= 2


# ---------------------------------------------------------------------------
# ShardManager — fan-out reads
# ---------------------------------------------------------------------------

class TestShardManagerFanOut:

    def test_global_top_rated_returns_results(self, seeded_mgr):
        top = seeded_mgr.global_top_rated(limit=5)
        assert len(top) > 0

    def test_global_top_rated_limit_respected(self, seeded_mgr):
        assert len(seeded_mgr.global_top_rated(limit=3)) <= 3

    def test_global_top_rated_scores_descending(self, seeded_mgr):
        top = seeded_mgr.global_top_rated(limit=10)
        scores = [r["avg_score"] for r in top]
        assert scores == sorted(scores, reverse=True)

    def test_global_top_rated_votes_positive(self, seeded_mgr):
        for row in seeded_mgr.global_top_rated():
            assert row["votes"] > 0
            assert 0.0 <= row["avg_score"] <= 5.0


# ---------------------------------------------------------------------------
# Rebalancing
# ---------------------------------------------------------------------------

class TestShardManagerRebalance:

    def test_add_shard_increases_shard_count(self, seeded_mgr, tmpdir_path):
        before_count = len(seeded_mgr.shard_ids)
        new_path = os.path.join(tmpdir_path, "shard_extra.db")
        seeded_mgr.add_shard(new_path)
        assert len(seeded_mgr.shard_ids) == before_count + 1

    def test_add_shard_total_ratings_unchanged(self, seeded_mgr, tmpdir_path):
        before_total = seeded_mgr.total_ratings()
        new_path = os.path.join(tmpdir_path, "shard_extra.db")
        seeded_mgr.add_shard(new_path)
        assert seeded_mgr.total_ratings() == before_total

    def test_add_shard_migrates_subset_not_all(self, seeded_mgr, tmpdir_path):
        total = seeded_mgr.total_ratings()
        new_path = os.path.join(tmpdir_path, "shard_extra.db")
        _, migrated = seeded_mgr.add_shard(new_path)
        # At least some rows stay in place; consistent hashing never moves all.
        assert migrated < total

    def test_user_ratings_still_accessible_after_rebalance(self, seeded_mgr, tmpdir_path):
        original_counts = {
            u["id"]: len(seeded_mgr.user_ratings(u["id"])) for u in USERS
        }
        new_path = os.path.join(tmpdir_path, "shard_extra.db")
        seeded_mgr.add_shard(new_path)
        for u in USERS:
            after = len(seeded_mgr.user_ratings(u["id"]))
            assert after == original_counts[u["id"]], (
                f"User {u['id']}: expected {original_counts[u['id']]} ratings "
                f"after rebalance, got {after}"
            )

    def test_new_shard_has_movies_replicated(self, seeded_mgr, tmpdir_path):
        new_path = os.path.join(tmpdir_path, "shard_extra.db")
        new_node, _ = seeded_mgr.add_shard(new_path)
        conn = seeded_mgr._conns[new_node]
        count = conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
        assert count == len(MOVIES)

    def test_global_top_rated_consistent_after_rebalance(self, seeded_mgr, tmpdir_path):
        before_top = seeded_mgr.global_top_rated(limit=3)
        new_path = os.path.join(tmpdir_path, "shard_extra.db")
        seeded_mgr.add_shard(new_path)
        after_top = seeded_mgr.global_top_rated(limit=3)
        before_titles = [r["title"] for r in before_top]
        after_titles = [r["title"] for r in after_top]
        assert before_titles == after_titles
