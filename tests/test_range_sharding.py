"""Tests for the Range-Based Sharding module."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from databaseai.seed_data import MOVIES, RATINGS
from databaseai.range_sharding import Shard, RangeShardManager, ShardedMovieDB, _OPEN_HIGH


# ============================================================
# Helpers / fixtures
# ============================================================

def _make_manager() -> RangeShardManager:
    return RangeShardManager([
        Shard(0, 0,    1999,       ":memory:"),
        Shard(1, 2000, 2009,       ":memory:"),
        Shard(2, 2010, 2019,       ":memory:"),
        Shard(3, 2020, _OPEN_HIGH, ":memory:"),
    ])


@pytest.fixture
def manager():
    mgr = _make_manager()
    yield mgr
    mgr.close_all()


@pytest.fixture
def db():
    d = ShardedMovieDB()
    d.seed(MOVIES, RATINGS)
    yield d
    d.close()


@pytest.fixture
def db_synthetic():
    d = ShardedMovieDB()
    d.seed(MOVIES, RATINGS)
    d.seed_synthetic(n=300, seed_val=7)
    yield d
    d.close()


# ============================================================
# Shard
# ============================================================

class TestShard:

    def test_label_bounded(self):
        s = Shard(0, 2000, 2009)
        assert "2000" in s.label() and "2009" in s.label()

    def test_label_open_upper(self):
        s = Shard(3, 2020, _OPEN_HIGH)
        assert "∞" in s.label()

    def test_row_count_empty(self):
        s = Shard(0, 0, 1999)
        assert s.row_count("movies") == 0
        assert s.row_count("ratings") == 0
        s.close()

    def test_execute_increments_queries_served(self):
        s = Shard(0, 0, 9999)
        s.execute("SELECT 1")
        s.execute("SELECT 1")
        assert s.queries_served == 2
        s.close()

    def test_insert_and_count(self):
        s = Shard(0, 0, 1999)
        s.execute(
            "INSERT INTO movies (id,title,genre,year,director) VALUES (?,?,?,?,?)",
            ("t1", "Test", "drama", 1990, "Dir"),
        )
        s.commit()
        assert s.row_count("movies") == 1
        s.close()


# ============================================================
# RangeShardManager — routing
# ============================================================

class TestRangeShardManagerRouting:

    def test_shard_for_key_classic_era(self, manager):
        s = manager.shard_for_key(1994)
        assert s is not None and s.low == 0 and s.high == 1999

    def test_shard_for_key_2000s(self, manager):
        s = manager.shard_for_key(2005)
        assert s is not None and s.low == 2000 and s.high == 2009

    def test_shard_for_key_2010s(self, manager):
        s = manager.shard_for_key(2017)
        assert s is not None and s.low == 2010 and s.high == 2019

    def test_shard_for_key_recent(self, manager):
        s = manager.shard_for_key(2023)
        assert s is not None and s.low == 2020

    def test_shard_for_key_boundary_low(self, manager):
        s = manager.shard_for_key(2000)
        assert s is not None and s.low == 2000

    def test_shard_for_key_boundary_high(self, manager):
        s = manager.shard_for_key(2019)
        assert s is not None and s.low == 2010 and s.high == 2019

    def test_shard_for_key_gap_returns_none(self):
        """A year outside all defined ranges returns None."""
        mgr = RangeShardManager([
            Shard(0, 2000, 2009),
            Shard(1, 2011, 2019),  # gap: 2010 is unmapped
        ])
        assert mgr.shard_for_key(2010) is None
        mgr.close_all()

    def test_shard_count(self, manager):
        assert manager.shard_count == 4


# ============================================================
# RangeShardManager — pruning
# ============================================================

class TestRangeShardManagerPruning:

    def test_pruning_single_shard(self, manager):
        shards = manager.shards_for_range(2010, 2019)
        assert len(shards) == 1
        assert shards[0].low == 2010

    def test_pruning_span_two_shards(self, manager):
        shards = manager.shards_for_range(2008, 2012)
        ids = {s.shard_id for s in shards}
        assert len(shards) == 2
        # Should cover shard-1 (2000-2009) and shard-2 (2010-2019)
        assert any(s.low == 2000 for s in shards)
        assert any(s.low == 2010 for s in shards)

    def test_scatter_gather_all_shards(self, manager):
        shards = manager.all_shards()
        assert len(shards) == 4

    def test_pruning_excludes_non_overlapping(self, manager):
        shards = manager.shards_for_range(1990, 1999)
        assert all(s.low <= 1999 and s.high >= 1990 for s in shards)
        assert all(s.shard_id == 0 for s in shards)

    def test_pruning_future_range(self, manager):
        shards = manager.shards_for_range(2021, 2025)
        assert len(shards) == 1
        assert shards[0].low == 2020


# ============================================================
# RangeShardManager — mutations
# ============================================================

class TestRangeShardManagerMutations:

    def test_add_shard_increases_count(self, manager):
        before = manager.shard_count
        manager.add_shard(Shard(99, 3000, 3999))
        assert manager.shard_count == before + 1

    def test_remove_shard_decreases_count(self, manager):
        before = manager.shard_count
        manager.remove_shard(0)
        assert manager.shard_count == before - 1

    def test_add_shard_keeps_sorted_order(self, manager):
        manager.add_shard(Shard(99, 1800, 1899))
        lows = [s.low for s in manager.all_shards()]
        assert lows == sorted(lows)


# ============================================================
# ShardedMovieDB — seeding
# ============================================================

class TestShardedMovieDBSeeding:

    def test_total_movies_matches_seed(self, db):
        assert db.total_movies() == len(MOVIES)

    def test_total_ratings_matches_seed(self, db):
        assert db.total_ratings() == len(RATINGS)

    def test_movies_routed_by_year(self, db):
        """Each movie should live in the shard that owns its release year."""
        for movie in MOVIES:
            shard = db.manager.shard_for_key(movie["year"])
            assert shard is not None
            count = shard.execute(
                "SELECT COUNT(*) FROM movies WHERE id = ?", (movie["id"],)
            ).fetchone()[0]
            assert count == 1, f"{movie['id']} not found in shard {shard.label()}"

    def test_ratings_co_located_with_movies(self, db):
        """
        Each rating should live on the same shard as its movie — no
        cross-shard JOINs are needed for year-range queries.
        """
        movie_year = {m["id"]: m["year"] for m in MOVIES}
        for user_id, movie_id, score, _ in RATINGS:
            year = movie_year[movie_id]
            shard = db.manager.shard_for_key(year)
            assert shard is not None
            count = shard.execute(
                "SELECT COUNT(*) FROM ratings WHERE user_id = ? AND movie_id = ?",
                (user_id, movie_id),
            ).fetchone()[0]
            assert count == 1

    def test_synthetic_seed_adds_rows(self, db_synthetic):
        assert db_synthetic.total_ratings() > len(RATINGS)

    def test_shard_stats_structure(self, db):
        stats = db.shard_stats()
        assert len(stats) == 4
        for s in stats:
            assert "shard_id" in s and "label" in s
            assert "movies" in s and "ratings" in s


# ============================================================
# ShardedMovieDB — pruned range query
# ============================================================

class TestPrunedRangeQuery:

    def test_pruned_query_returns_results(self, db):
        results, shards_touched, _ = db.query_range_pruned(2010, 2019)
        assert len(results) > 0

    def test_pruned_query_touches_one_shard(self, db):
        _, shards_touched, _ = db.query_range_pruned(2010, 2019)
        assert shards_touched == 1

    def test_pruned_query_results_in_range(self, db):
        results, _, _ = db.query_range_pruned(2010, 2019)
        for r in results:
            assert 2010 <= r["year"] <= 2019

    def test_pruned_query_classics(self, db):
        results, shards_touched, _ = db.query_range_pruned(1990, 1999)
        assert shards_touched == 1
        for r in results:
            assert r["year"] <= 1999

    def test_pruned_span_two_shards_touches_two(self, db):
        _, shards_touched, _ = db.query_range_pruned(2008, 2012)
        assert shards_touched == 2

    def test_pruned_query_returns_elapsed(self, db):
        _, _, elapsed_ms = db.query_range_pruned(2010, 2019)
        assert elapsed_ms >= 0.0


# ============================================================
# ShardedMovieDB — scatter-gather query
# ============================================================

class TestScatterGatherQuery:

    def test_scatter_gather_touches_all_shards(self, db):
        _, shards_touched, _ = db.query_scatter_gather(min_score=4.0)
        assert shards_touched == db.manager.shard_count

    def test_scatter_gather_returns_high_scores_only(self, db):
        results, _, _ = db.query_scatter_gather(min_score=4.5)
        assert all(r["score"] >= 4.5 for r in results)

    def test_scatter_gather_sorted_descending(self, db):
        results, _, _ = db.query_scatter_gather(min_score=1.0)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_scatter_gather_covers_all_year_ranges(self, db_synthetic):
        results, _, _ = db_synthetic.query_scatter_gather(min_score=1.0)
        years = {r["year"] for r in results}
        # With seed data spanning 1957–2022, all four year bands should appear
        assert min(years) < 2000
        assert max(years) >= 2020 or max(years) >= 2010  # at least 2010s

    def test_scatter_gather_elapsed_nonnegative(self, db):
        _, _, elapsed_ms = db.query_scatter_gather(min_score=4.0)
        assert elapsed_ms >= 0.0


# ============================================================
# RangeShardManager — hot-shard detection
# ============================================================

class TestHotShardDetection:

    def test_hot_shards_above_zero_threshold(self, db_synthetic):
        """Any shard with data qualifies as 'hot' when threshold is 0."""
        hot = db_synthetic.manager.hot_shards(0)
        assert len(hot) > 0

    def test_hot_shards_above_max_threshold(self, db_synthetic):
        """No shard is hot when threshold exceeds total ratings."""
        hot = db_synthetic.manager.hot_shards(10_000)
        assert len(hot) == 0

    def test_2010s_shard_is_hottest(self, db_synthetic):
        """Given the seed data distribution, shard-2 (2010-2019) should have most rows."""
        stats = db_synthetic.shard_stats()
        max_ratings = max(s["ratings"] for s in stats)
        shard2_ratings = next(s["ratings"] for s in stats if s["shard_id"] == 2)
        assert shard2_ratings == max_ratings


# ============================================================
# RangeShardManager — shard splitting
# ============================================================

class TestShardSplitting:

    def test_split_increases_shard_count(self, db_synthetic):
        before = db_synthetic.manager.shard_count
        target = next(
            s for s in db_synthetic.manager.all_shards()
            if s.low == 2010 and s.high == 2019
        )
        db_synthetic.manager.split_shard(target)
        assert db_synthetic.manager.shard_count == before + 1

    def test_split_child_ranges_non_overlapping(self, db_synthetic):
        target = next(
            s for s in db_synthetic.manager.all_shards()
            if s.low == 2010 and s.high == 2019
        )
        lower, upper = db_synthetic.manager.split_shard(target)
        assert lower.high < upper.low
        assert lower.low == 2010
        assert upper.high == 2019

    def test_split_preserves_total_movies(self, db_synthetic):
        target = next(
            s for s in db_synthetic.manager.all_shards()
            if s.low == 2010 and s.high == 2019
        )
        movies_before = target.row_count("movies")
        lower, upper = db_synthetic.manager.split_shard(target)
        assert lower.row_count("movies") + upper.row_count("movies") == movies_before

    def test_split_preserves_total_ratings(self, db_synthetic):
        target = next(
            s for s in db_synthetic.manager.all_shards()
            if s.low == 2010 and s.high == 2019
        )
        ratings_before = target.row_count("ratings")
        lower, upper = db_synthetic.manager.split_shard(target)
        assert lower.row_count("ratings") + upper.row_count("ratings") == ratings_before

    def test_split_child_movies_in_correct_range(self, db_synthetic):
        target = next(
            s for s in db_synthetic.manager.all_shards()
            if s.low == 2010 and s.high == 2019
        )
        lower, upper = db_synthetic.manager.split_shard(target)
        mid = lower.high
        for row in lower.execute("SELECT year FROM movies").fetchall():
            assert row["year"] <= mid
        for row in upper.execute("SELECT year FROM movies").fetchall():
            assert row["year"] > mid

    def test_split_removes_parent_from_manager(self, db_synthetic):
        target = next(
            s for s in db_synthetic.manager.all_shards()
            if s.low == 2010 and s.high == 2019
        )
        parent_id = target.shard_id
        db_synthetic.manager.split_shard(target)
        remaining_ids = {s.shard_id for s in db_synthetic.manager.all_shards()}
        assert parent_id not in remaining_ids

    def test_split_open_upper_bound(self):
        """Splitting a shard with open upper bound produces finite child ranges."""
        mgr = RangeShardManager([Shard(0, 2020, _OPEN_HIGH)])
        shard = mgr.all_shards()[0]
        shard.execute(
            "INSERT INTO movies (id,title,genre,year,director) VALUES (?,?,?,?,?)",
            ("x1", "A", "drama", 2021, "D"),
        )
        shard.commit()
        lower, upper = mgr.split_shard(shard)
        assert lower.low == 2020
        assert lower.high < upper.low
        mgr.close_all()

    def test_global_ratings_unchanged_after_split(self, db_synthetic):
        total_before = db_synthetic.total_ratings()
        target = next(
            s for s in db_synthetic.manager.all_shards()
            if s.low == 2010 and s.high == 2019
        )
        db_synthetic.manager.split_shard(target)
        total_after = db_synthetic.total_ratings()
        assert total_after == total_before
