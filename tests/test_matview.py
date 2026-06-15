"""Tests for the Materialized Views module (matview)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from databaseai.matview import MaterializedViewStore
from databaseai.seed_data import MOVIES, USERS, RATINGS


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def empty_store():
    """Empty in-memory store with schema but no data."""
    return MaterializedViewStore()


@pytest.fixture
def store():
    """Store seeded with full MOVIES + RATINGS data, both MVs refreshed."""
    s = MaterializedViewStore()
    s.load_seed(MOVIES, RATINGS)
    s.refresh_all()
    return s


@pytest.fixture
def stale_store():
    """Store seeded with data but MVs have never been refreshed (stale)."""
    s = MaterializedViewStore()
    s.load_seed(MOVIES, RATINGS)
    return s


# ── Construction & seed loading ───────────────────────────────────────────────

class TestConstruction:

    def test_movie_count_matches_seed(self, store):
        assert store.movie_count() == len(MOVIES)

    def test_rating_count_matches_seed(self, store):
        assert store.rating_count() == len(RATINGS)

    def test_empty_store_has_no_movies(self, empty_store):
        assert empty_store.movie_count() == 0

    def test_load_seed_idempotent(self):
        s = MaterializedViewStore()
        s.load_seed(MOVIES, RATINGS)
        s.load_seed(MOVIES, RATINGS)
        assert s.movie_count() == len(MOVIES)
        assert s.rating_count() == len(RATINGS)


# ── Genre stats MV ────────────────────────────────────────────────────────────

class TestGenreStats:

    def test_genre_stats_row_count(self, store):
        rows = store.get_genre_stats()
        genres_in_seed = {m["genre"] for m in MOVIES}
        assert len(rows) == len(genres_in_seed)

    def test_genre_stats_avg_rating_in_range(self, store):
        for row in store.get_genre_stats():
            assert 1.0 <= row["avg_rating"] <= 5.0

    def test_genre_stats_sorted_by_avg_desc(self, store):
        rows = store.get_genre_stats()
        ratings = [r["avg_rating"] for r in rows]
        assert ratings == sorted(ratings, reverse=True)

    def test_genre_stats_top_movie_is_not_none(self, store):
        rows = store.get_genre_stats()
        rated_genres = {r[1] for r in RATINGS for m in MOVIES if m["id"] == r[1]}
        for row in rows:
            if row["rating_count"] > 0:
                assert row["top_movie_title"] is not None

    def test_genre_stats_movie_count_correct_for_scifi(self, store):
        scifi_count = sum(1 for m in MOVIES if m["genre"] == "sci-fi")
        rows = store.get_genre_stats()
        scifi_row = next((r for r in rows if r["genre"] == "sci-fi"), None)
        assert scifi_row is not None
        assert scifi_row["movie_count"] == scifi_count


# ── Top movies MV ─────────────────────────────────────────────────────────────

class TestTopMovies:

    def test_top_movies_sorted_by_avg_desc(self, store):
        rows = store.get_top_movies(n=20)
        ratings = [r["avg_rating"] for r in rows]
        assert ratings == sorted(ratings, reverse=True)

    def test_top_movies_rank_is_sequential(self, store):
        rows = store.get_top_movies(n=10)
        for i, row in enumerate(rows, 1):
            assert row["rank"] == i

    def test_top_movies_n_limit_respected(self, store):
        for n in [1, 3, 5, 10]:
            assert len(store.get_top_movies(n=n)) <= n

    def test_top_movies_each_movie_has_required_fields(self, store):
        for row in store.get_top_movies(n=10):
            assert "movie_id" in row
            assert "title" in row
            assert "genre" in row
            assert "avg_rating" in row
            assert "rating_count" in row


# ── Staleness & refresh ───────────────────────────────────────────────────────

class TestStaleness:

    def test_is_stale_initially_true(self, stale_store):
        assert stale_store.is_stale(MaterializedViewStore.VIEW_GENRE_STATS)
        assert stale_store.is_stale(MaterializedViewStore.VIEW_TOP_MOVIES)

    def test_is_stale_false_after_refresh(self, store):
        assert not store.is_stale(MaterializedViewStore.VIEW_GENRE_STATS)
        assert not store.is_stale(MaterializedViewStore.VIEW_TOP_MOVIES)

    def test_mark_stale_sets_flag(self, store):
        assert not store.is_stale(MaterializedViewStore.VIEW_GENRE_STATS)
        store.mark_stale(MaterializedViewStore.VIEW_GENRE_STATS)
        assert store.is_stale(MaterializedViewStore.VIEW_GENRE_STATS)

    def test_mark_stale_all_views(self, store):
        store.mark_stale()
        assert store.is_stale(MaterializedViewStore.VIEW_GENRE_STATS)
        assert store.is_stale(MaterializedViewStore.VIEW_TOP_MOVIES)

    def test_add_rating_marks_views_stale(self, store):
        store.add_rating("u05", "m02", 4.0, eager=False)
        assert store.is_stale(MaterializedViewStore.VIEW_GENRE_STATS)
        assert store.is_stale(MaterializedViewStore.VIEW_TOP_MOVIES)

    def test_add_rating_eager_leaves_views_fresh(self, store):
        store.add_rating("u05", "m02", 4.0, eager=True)
        assert not store.is_stale(MaterializedViewStore.VIEW_GENRE_STATS)
        assert not store.is_stale(MaterializedViewStore.VIEW_TOP_MOVIES)

    def test_lazy_read_refreshes_stale_view(self, store):
        store.mark_stale(MaterializedViewStore.VIEW_GENRE_STATS)
        assert store.is_stale(MaterializedViewStore.VIEW_GENRE_STATS)
        rows = store.get_genre_stats(lazy=True)
        assert len(rows) > 0
        assert not store.is_stale(MaterializedViewStore.VIEW_GENRE_STATS)

    def test_non_lazy_read_does_not_refresh(self, store):
        store.mark_stale(MaterializedViewStore.VIEW_GENRE_STATS)
        store.get_genre_stats(lazy=False)
        assert store.is_stale(MaterializedViewStore.VIEW_GENRE_STATS)


# ── Refresh metadata ──────────────────────────────────────────────────────────

class TestRefreshMeta:

    def test_refresh_count_increments(self, stale_store):
        before = stale_store.get_meta(MaterializedViewStore.VIEW_GENRE_STATS)
        stale_store.refresh_genre_stats()
        after = stale_store.get_meta(MaterializedViewStore.VIEW_GENRE_STATS)
        assert after["refresh_count"] == before["refresh_count"] + 1

    def test_refresh_returns_positive_ms(self, stale_store):
        ms = stale_store.refresh_genre_stats()
        assert ms >= 0.0

    def test_refresh_all_returns_both_views(self, stale_store):
        result = stale_store.refresh_all()
        assert MaterializedViewStore.VIEW_GENRE_STATS in result
        assert MaterializedViewStore.VIEW_TOP_MOVIES in result

    def test_total_refresh_ms_accumulates(self, stale_store):
        stale_store.refresh_genre_stats()
        meta1 = stale_store.get_meta(MaterializedViewStore.VIEW_GENRE_STATS)
        stale_store.mark_stale(MaterializedViewStore.VIEW_GENRE_STATS)
        stale_store.refresh_genre_stats()
        meta2 = stale_store.get_meta(MaterializedViewStore.VIEW_GENRE_STATS)
        assert meta2["total_refresh_ms"] >= meta1["total_refresh_ms"]


# ── Incremental refresh ───────────────────────────────────────────────────────

class TestIncrementalRefresh:

    def test_incremental_refresh_updates_target_genre(self, store):
        before = next(r for r in store.get_genre_stats() if r["genre"] == "sci-fi")
        store.add_rating("u05", "m04", 2.0, eager=False)
        store.refresh_genre_for(["sci-fi"])
        after = next(r for r in store.get_genre_stats() if r["genre"] == "sci-fi")
        assert after["rating_count"] > before["rating_count"]

    def test_incremental_refresh_empty_genres_noop(self, store):
        elapsed = store.refresh_genre_for([])
        assert elapsed == 0.0

    def test_incremental_refresh_returns_positive_ms_for_valid_genre(self, store):
        ms = store.refresh_genre_for(["sci-fi"])
        assert ms >= 0.0

    def test_incremental_refresh_does_not_affect_other_genres(self, store):
        drama_before = next(
            (r for r in store.get_genre_stats() if r["genre"] == "drama"), None
        )
        store.refresh_genre_for(["sci-fi"])
        drama_after = next(
            (r for r in store.get_genre_stats() if r["genre"] == "drama"), None
        )
        if drama_before and drama_after:
            assert drama_before["avg_rating"] == drama_after["avg_rating"]


# ── Benchmark ─────────────────────────────────────────────────────────────────

class TestBenchmark:

    def test_benchmark_returns_all_keys(self, store):
        result = store.benchmark(n_queries=10)
        for key in ("n_queries", "live_total_ms", "mv_total_ms",
                    "live_avg_ms", "mv_avg_ms", "speedup_x"):
            assert key in result

    def test_benchmark_n_queries_matches_request(self, store):
        result = store.benchmark(n_queries=5)
        assert result["n_queries"] == 5

    def test_benchmark_timings_are_positive(self, store):
        result = store.benchmark(n_queries=10)
        assert result["live_total_ms"] >= 0.0
        assert result["mv_total_ms"] >= 0.0

    def test_benchmark_speedup_is_positive(self, store):
        result = store.benchmark(n_queries=10)
        assert result["speedup_x"] > 0.0


# ── Live vs MV consistency ────────────────────────────────────────────────────

class TestLiveVsMV:

    def test_live_genre_stats_same_genres_as_mv(self, store):
        live = {r["genre"] for r in store.live_genre_stats()}
        mv   = {r["genre"] for r in store.get_genre_stats()}
        assert live == mv

    def test_live_top_movies_same_top_as_mv(self, store):
        live_top = store.live_top_movies(n=1)
        mv_top   = store.get_top_movies(n=1)
        assert len(live_top) == 1
        assert len(mv_top)   == 1
        assert live_top[0]["movie_id"] == mv_top[0]["movie_id"]

    def test_add_rating_changes_live_result_immediately(self, store):
        before = store.live_top_movies(n=20)
        store.add_rating("u01", "m19", 5.0, eager=False)
        after = store.live_top_movies(n=20)
        m19_before = next((r for r in before if r["movie_id"] == "m19"), None)
        m19_after  = next((r for r in after  if r["movie_id"] == "m19"), None)
        assert m19_after is not None
        if m19_before:
            assert m19_after["rating_count"] > m19_before["rating_count"]
