"""Tests for the Content Performance Analytics module (Demo 15)."""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datetime import date, timedelta

from databaseai.content_analytics import ContentMetricsStore
from databaseai.seed_data import MOVIES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LAUNCH = date(2024, 11, 1)
DAYS = 10


def _d(offset: int) -> str:
    return (LAUNCH + timedelta(days=offset)).strftime("%Y-%m-%d")


def _build_rows(movie_ids, days=DAYS):
    """
    Deterministic synthetic rows.  Movie at list index i gets base views of
    (i+1)*1000 decaying by 10 per day, so totals and rankings are predictable.
    """
    rows = []
    for i, mid in enumerate(movie_ids):
        for d in range(days):
            views = max((i + 1) * 1000 - d * 10, 1)
            completions = max(views // 2, 1)
            ratings = max(views // 50, 1)
            rows.append((mid, _d(d), views, completions, ratings, 70.0))
    return rows


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    s = ContentMetricsStore(db_path=str(tmp_path / "test_ca.db"))
    yield s
    s.close()


@pytest.fixture
def seeded_store(tmp_path):
    s = ContentMetricsStore(db_path=str(tmp_path / "test_ca_seeded.db"))
    rows = _build_rows(["m01", "m02", "m05", "m07"])
    s.bulk_record_metrics(rows)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# 1. Ingestion
# ---------------------------------------------------------------------------

class TestIngestion:

    def test_metric_count_after_bulk_insert(self, store):
        rows = _build_rows(["m01", "m05"], days=5)
        store.bulk_record_metrics(rows)
        assert store.metric_count() == 10  # 2 movies x 5 days

    def test_bulk_insert_is_idempotent(self, store):
        rows = _build_rows(["m01"], days=5)
        store.bulk_record_metrics(rows)
        store.bulk_record_metrics(rows)  # second insert should replace, not duplicate
        assert store.metric_count() == 5

    def test_event_count_after_records(self, store):
        store.record_event("m01", "launch", "Movie launched", {}, _d(0))
        store.record_event("m01", "award",  "Won an award",   {}, _d(5))
        assert store.event_count() == 2

    def test_record_event_with_none_movie_id(self, store):
        store.record_event(None, "system", "Platform-wide update", {"v": 2}, _d(1))
        assert store.event_count() == 1

    def test_empty_store_has_zero_metric_count(self, store):
        assert store.metric_count() == 0

    def test_empty_store_has_zero_event_count(self, store):
        assert store.event_count() == 0


# ---------------------------------------------------------------------------
# 2. Time-windowed queries
# ---------------------------------------------------------------------------

class TestTimeWindowedQueries:

    def test_query_window_returns_correct_row_count(self, seeded_store):
        rows = seeded_store.query_window("m01", _d(0), _d(4))
        assert len(rows) == 5  # D0 through D4 inclusive

    def test_query_window_excludes_out_of_range_dates(self, seeded_store):
        rows = seeded_store.query_window("m01", _d(2), _d(5))
        for r in rows:
            assert _d(2) <= r["event_date"] <= _d(5)

    def test_query_window_ordered_ascending(self, seeded_store):
        rows = seeded_store.query_window("m01", _d(0), _d(DAYS - 1))
        dates = [r["event_date"] for r in rows]
        assert dates == sorted(dates)

    def test_aggregate_window_day_count(self, seeded_store):
        agg = seeded_store.aggregate_window("m01", _d(0), _d(4))
        assert agg["days"] == 5

    def test_aggregate_window_total_views_matches_sum(self, seeded_store):
        agg = seeded_store.aggregate_window("m01", _d(0), _d(DAYS - 1))
        # m01 at index 0: views = max(1000 - d*10, 1) for d in range(DAYS)
        expected = sum(max(1000 - d * 10, 1) for d in range(DAYS))
        assert agg["total_views"] == expected

    def test_aggregate_window_returns_none_for_unknown_movie(self, seeded_store):
        agg = seeded_store.aggregate_window("m99", _d(0), _d(DAYS - 1))
        assert agg["total_views"] is None


# ---------------------------------------------------------------------------
# 3. Rolling averages
# ---------------------------------------------------------------------------

class TestRollingAverage:

    def test_rolling_average_length_equals_total_days(self, seeded_store):
        result = seeded_store.rolling_average("m01", "views", 7)
        assert len(result) == DAYS

    def test_rolling_average_day0_equals_raw_value(self, seeded_store):
        result = seeded_store.rolling_average("m01", "views", 7)
        assert result[0]["rolling_avg"] == result[0]["raw_value"]

    def test_rolling_average_window_3_matches_manual_calculation(self, seeded_store):
        result = seeded_store.rolling_average("m01", "views", 3)
        # Day 2 rolling avg = mean of days 0, 1, 2
        rows = seeded_store.query_window("m01", _d(0), _d(2))
        expected = sum(r["views"] for r in rows) / 3
        assert abs(result[2]["rolling_avg"] - expected) < 0.01

    def test_rolling_average_invalid_metric_raises_value_error(self, seeded_store):
        with pytest.raises(ValueError):
            seeded_store.rolling_average("m01", "bad_column", 7)

    def test_rolling_average_contains_required_keys(self, seeded_store):
        result = seeded_store.rolling_average("m01", "views", 3)
        required = {"event_date", "raw_value", "rolling_avg"}
        assert required.issubset(result[0].keys())


# ---------------------------------------------------------------------------
# 4. Retention curves
# ---------------------------------------------------------------------------

class TestRetentionCurve:

    def test_retention_curve_length_equals_total_days(self, seeded_store):
        curve = seeded_store.retention_curve("m01")
        assert len(curve) == DAYS

    def test_retention_curve_day0_relative_views_is_1(self, seeded_store):
        curve = seeded_store.retention_curve("m01")
        assert curve[0]["relative_views"] == 1.0

    def test_retention_curve_declines_for_monotone_decay(self, seeded_store):
        curve = seeded_store.retention_curve("m01")
        assert curve[-1]["relative_views"] < curve[0]["relative_views"]

    def test_retention_curve_empty_for_unknown_movie(self, seeded_store):
        assert seeded_store.retention_curve("m99") == []

    def test_retention_curve_contains_required_keys(self, seeded_store):
        curve = seeded_store.retention_curve("m01")
        required = {"days_since_launch", "event_date", "views", "relative_views"}
        assert required.issubset(curve[0].keys())

    def test_retention_curve_days_since_launch_sequential(self, seeded_store):
        curve = seeded_store.retention_curve("m01")
        for i, entry in enumerate(curve):
            assert entry["days_since_launch"] == i


# ---------------------------------------------------------------------------
# 5. Peak detection
# ---------------------------------------------------------------------------

class TestPeakDetection:

    def test_peak_day_is_day0_for_strictly_decaying_series(self, seeded_store):
        peak = seeded_store.peak_day("m01", "views")
        assert peak["event_date"] == _d(0)

    def test_peak_day_returns_correct_movie_id(self, seeded_store):
        peak = seeded_store.peak_day("m01", "views")
        assert peak["movie_id"] == "m01"

    def test_peak_day_returns_empty_dict_for_unknown_movie(self, seeded_store):
        assert seeded_store.peak_day("m99", "views") == {}

    def test_peak_day_invalid_metric_raises_value_error(self, seeded_store):
        with pytest.raises(ValueError):
            seeded_store.peak_day("m01", "not_a_real_column")

    def test_peak_day_completions_metric(self, seeded_store):
        peak = seeded_store.peak_day("m02", "completions")
        assert "peak_value" in peak
        assert peak["peak_value"] > 0


# ---------------------------------------------------------------------------
# 6. Genre trends
# ---------------------------------------------------------------------------

class TestGenreTrends:

    def test_genre_trend_returns_one_row_per_day(self, seeded_store):
        movie_genre_map = {m["id"]: m["genre"] for m in MOVIES}
        trend = seeded_store.genre_trend(
            "sci-fi", "views", _d(0), _d(DAYS - 1), movie_genre_map
        )
        assert len(trend) == DAYS

    def test_genre_trend_sums_across_multiple_movies(self, seeded_store):
        # Only m01 and m02 are seeded; force both to the same genre for this test
        movie_genre_map = {"m01": "sci-fi", "m02": "sci-fi"}
        trend = seeded_store.genre_trend(
            "sci-fi", "views", _d(0), _d(0), movie_genre_map
        )
        assert len(trend) == 1
        # m01 day0 = 1000, m02 day0 = 2000 -> total = 3000
        assert trend[0]["total_value"] == 3000

    def test_genre_trend_empty_for_unrepresented_genre(self, seeded_store):
        movie_genre_map = {m["id"]: m["genre"] for m in MOVIES}
        trend = seeded_store.genre_trend(
            "fantasy", "views", _d(0), _d(DAYS - 1), movie_genre_map
        )
        assert trend == []

    def test_genre_trend_invalid_metric_raises_value_error(self, seeded_store):
        movie_genre_map = {m["id"]: m["genre"] for m in MOVIES}
        with pytest.raises(ValueError):
            seeded_store.genre_trend(
                "sci-fi", "avg_watch_pct", _d(0), _d(DAYS - 1), movie_genre_map
            )

    def test_genre_trend_contains_required_keys(self, seeded_store):
        movie_genre_map = {m["id"]: m["genre"] for m in MOVIES}
        trend = seeded_store.genre_trend(
            "sci-fi", "views", _d(0), _d(DAYS - 1), movie_genre_map
        )
        required = {"event_date", "total_value", "movie_count"}
        assert required.issubset(trend[0].keys())


# ---------------------------------------------------------------------------
# 7. Top movies
# ---------------------------------------------------------------------------

class TestTopMovies:

    def test_top_movies_returns_at_most_n_results(self, seeded_store):
        top = seeded_store.top_movies_in_window(_d(0), _d(DAYS - 1), "views", 3)
        assert len(top) == 3

    def test_top_movies_ordered_descending_by_total(self, seeded_store):
        top = seeded_store.top_movies_in_window(_d(0), _d(DAYS - 1), "views", 4)
        values = [r["total_value"] for r in top]
        assert values == sorted(values, reverse=True)

    def test_top_movies_highest_base_movie_ranks_first(self, seeded_store):
        # m07 at index 3 has base 4000, so it accumulates the most views
        top = seeded_store.top_movies_in_window(_d(0), _d(DAYS - 1), "views", 4)
        assert top[0]["movie_id"] == "m07"

    def test_top_movies_invalid_metric_raises_value_error(self, seeded_store):
        with pytest.raises(ValueError):
            seeded_store.top_movies_in_window(_d(0), _d(DAYS - 1), "avg_watch_pct", 3)

    def test_top_movies_result_contains_movie_id_and_total_value(self, seeded_store):
        top = seeded_store.top_movies_in_window(_d(0), _d(DAYS - 1), "views", 2)
        for row in top:
            assert "movie_id" in row
            assert "total_value" in row


# ---------------------------------------------------------------------------
# 8. Events
# ---------------------------------------------------------------------------

class TestEvents:

    def test_query_events_returns_all_in_window(self, store):
        store.record_event("m01", "launch", "Launched",  {}, _d(0))
        store.record_event("m05", "award",  "Won award", {}, _d(5))
        store.record_event("m09", "press",  "Press hit", {}, _d(8))
        events = store.query_events(_d(0), _d(DAYS))
        assert len(events) == 3

    def test_query_events_excludes_out_of_window(self, store):
        store.record_event("m01", "launch", "Launched",  {}, _d(0))
        store.record_event("m05", "award",  "Old event", {}, _d(15))  # outside window
        events = store.query_events(_d(0), _d(10))
        assert len(events) == 1

    def test_query_events_filtered_by_movie_id(self, store):
        store.record_event("m01", "launch", "m01 launched", {}, _d(0))
        store.record_event("m01", "spike",  "m01 spike",    {}, _d(3))
        store.record_event("m05", "award",  "m05 award",    {}, _d(7))
        events = store.query_events(_d(0), _d(DAYS), movie_id="m01")
        assert len(events) == 2
        assert all(e["movie_id"] == "m01" for e in events)

    def test_query_events_ordered_chronologically(self, store):
        store.record_event("m01", "c", "Third",  {}, _d(8))
        store.record_event("m01", "a", "First",  {}, _d(2))
        store.record_event("m01", "b", "Second", {}, _d(5))
        events = store.query_events(_d(0), _d(DAYS))
        dates = [e["occurred_at"] for e in events]
        assert dates == sorted(dates)

    def test_platform_event_with_none_movie_id_is_returned(self, store):
        store.record_event(None, "system", "Algo update", {"v": 2}, _d(3))
        events = store.query_events(_d(0), _d(DAYS))
        assert len(events) == 1
        assert events[0]["movie_id"] is None
