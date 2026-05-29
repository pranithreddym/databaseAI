"""Tests for the Engagement Time-Series Store (engagement_ts module)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from databaseai.engagement_ts import EngagementStore


# -- Fixtures

@pytest.fixture
def store():
    """
    Two-user x three-movie store with deterministic sessions and events:
      Sessions:
        u01 m01  2024-03-01 09:00  7800 s  completed
        u01 m02  2024-03-01 20:00  5400 s  dropped
        u01 m03  2024-03-02 20:00  9000 s  completed
        u01 m01  2024-03-07 21:00  8280 s  completed
        u02 m01  2024-03-01 21:00  7200 s  completed
        u02 m03  2024-03-03 19:00  4800 s  dropped
        u02 m03  2024-03-08 20:00  9600 s  completed
      Events:
        u01 play        2024-03-01 09:00
        u01 pause       2024-03-01 09:30
        u01 play        2024-03-01 20:00
        u02 play        2024-03-01 21:00
        u02 search      2024-03-01 21:05
        u01 play        2024-03-02 20:00
        u01 add_to_list 2024-03-02 21:00
    """
    s = EngagementStore()
    s.bulk_record_sessions([
        ("u01", "m01", "2024-03-01 09:00:00", 7800, True),
        ("u01", "m02", "2024-03-01 20:00:00", 5400, False),
        ("u01", "m03", "2024-03-02 20:00:00", 9000, True),
        ("u01", "m01", "2024-03-07 21:00:00", 8280, True),
        ("u02", "m01", "2024-03-01 21:00:00", 7200, True),
        ("u02", "m03", "2024-03-03 19:00:00", 4800, False),
        ("u02", "m03", "2024-03-08 20:00:00", 9600, True),
    ])
    s.bulk_record_events([
        ("u01", "play",        "2024-03-01 09:00:00", {}),
        ("u01", "pause",       "2024-03-01 09:30:00", {}),
        ("u01", "play",        "2024-03-01 20:00:00", {}),
        ("u02", "play",        "2024-03-01 21:00:00", {}),
        ("u02", "search",      "2024-03-01 21:05:00", {"query": "action"}),
        ("u01", "play",        "2024-03-02 20:00:00", {}),
        ("u01", "add_to_list", "2024-03-02 21:00:00", {"movie_id": "m03"}),
    ])
    return s


_FULL_START = "2024-03-01 00:00:00"
_FULL_END   = "2024-03-09 23:59:59"


# -- Construction & counts

class TestCounts:

    def test_session_count(self, store):
        assert store.session_count() == 7

    def test_event_count(self, store):
        assert store.event_count() == 7

    def test_empty_store_returns_zero_session_count(self):
        assert EngagementStore().session_count() == 0

    def test_empty_store_returns_zero_event_count(self):
        assert EngagementStore().event_count() == 0

    def test_record_session_increments_count(self):
        s = EngagementStore()
        s.record_session("u1", "m1", "2024-01-01 10:00:00", 3600, True)
        assert s.session_count() == 1

    def test_record_event_increments_count(self):
        s = EngagementStore()
        s.record_event("u1", "play", "2024-01-01 10:00:00")
        assert s.event_count() == 1


# -- Rolling window stats

class TestRollingWindowStats:

    def test_full_window_session_count(self, store):
        st = store.rolling_window_stats(_FULL_START, _FULL_END)
        assert st["session_count"] == 7

    def test_full_window_unique_users(self, store):
        st = store.rolling_window_stats(_FULL_START, _FULL_END)
        assert st["unique_users"] == 2

    def test_completion_pct_is_correct(self, store):
        # 5 completed out of 7 total -> 71.4 %
        st = store.rolling_window_stats(_FULL_START, _FULL_END)
        assert abs(st["completion_pct"] - 71.4) < 0.2

    def test_narrow_window_excludes_later_sessions(self, store):
        # Only 3 sessions on 2024-03-01
        st = store.rolling_window_stats(
            "2024-03-01 00:00:00", "2024-03-01 23:59:59"
        )
        assert st["session_count"] == 3

    def test_empty_window_returns_zero_count(self, store):
        st = store.rolling_window_stats(
            "2030-01-01 00:00:00", "2030-01-01 23:59:59"
        )
        assert st["session_count"] == 0

    def test_avg_watch_min_is_positive(self, store):
        st = store.rolling_window_stats(_FULL_START, _FULL_END)
        assert st["avg_watch_min"] > 0

    def test_max_watch_min_gte_avg(self, store):
        st = store.rolling_window_stats(_FULL_START, _FULL_END)
        assert st["max_watch_min"] >= st["avg_watch_min"]


# -- Hourly activity

class TestHourlyActivity:

    def test_returns_one_row_per_active_hour(self, store):
        # 7 sessions spread across 7 distinct hour-buckets
        rows = store.hourly_activity(_FULL_START, _FULL_END)
        assert len(rows) == 7

    def test_rows_sorted_chronologically(self, store):
        rows = store.hourly_activity(_FULL_START, _FULL_END)
        buckets = [r["hour_bucket"] for r in rows]
        assert buckets == sorted(buckets)

    def test_correct_hour_bucket_format(self, store):
        rows = store.hourly_activity(_FULL_START, _FULL_END)
        assert all(r["hour_bucket"].endswith(":00") for r in rows)

    def test_sessions_per_bucket_sum_equals_total(self, store):
        rows = store.hourly_activity(_FULL_START, _FULL_END)
        assert sum(r["sessions"] for r in rows) == store.session_count()


# -- Peak hours

class TestPeakHours:

    def test_top_hour_has_most_sessions(self, store):
        # Hour 20 has 3 sessions (days 1, 2, 8)
        peaks = store.peak_hours(_FULL_START, _FULL_END, top_n=5)
        assert peaks[0]["hour_of_day"] == 20
        assert peaks[0]["session_count"] == 3

    def test_results_sorted_descending_by_session_count(self, store):
        peaks = store.peak_hours(_FULL_START, _FULL_END, top_n=5)
        counts = [r["session_count"] for r in peaks]
        assert counts == sorted(counts, reverse=True)

    def test_top_n_limit_respected(self, store):
        for n in [1, 2, 3]:
            peaks = store.peak_hours(_FULL_START, _FULL_END, top_n=n)
            assert len(peaks) <= n

    def test_hour_of_day_in_valid_range(self, store):
        peaks = store.peak_hours(_FULL_START, _FULL_END, top_n=24)
        assert all(0 <= r["hour_of_day"] <= 23 for r in peaks)


# -- Daily active users

class TestDailyActiveUsers:

    def test_correct_number_of_active_days(self, store):
        # 5 distinct days have sessions: 03-01, 03-02, 03-03, 03-07, 03-08
        rows = store.daily_active_users(_FULL_START, _FULL_END)
        assert len(rows) == 5

    def test_day_with_two_users_has_dau_two(self, store):
        rows = store.daily_active_users(_FULL_START, _FULL_END)
        day1 = next(r for r in rows if r["day"] == "2024-03-01")
        assert day1["dau"] == 2

    def test_single_user_day_has_dau_one(self, store):
        rows = store.daily_active_users(_FULL_START, _FULL_END)
        day2 = next(r for r in rows if r["day"] == "2024-03-02")
        assert day2["dau"] == 1

    def test_rows_sorted_chronologically(self, store):
        rows = store.daily_active_users(_FULL_START, _FULL_END)
        days = [r["day"] for r in rows]
        assert days == sorted(days)

    def test_sessions_column_matches_actual_count(self, store):
        rows = store.daily_active_users(_FULL_START, _FULL_END)
        # 2024-03-01 has 3 sessions (u01x2 + u02x1)
        day1 = next(r for r in rows if r["day"] == "2024-03-01")
        assert day1["sessions"] == 3


# -- Completion rate by movie

class TestCompletionRateByMovie:

    def test_m01_is_fully_completed(self, store):
        # m01: 3 sessions all completed -> 100 %
        rates = store.completion_rate_by_movie(["m01"])
        assert len(rates) == 1
        assert rates[0]["completion_pct"] == 100.0
        assert rates[0]["total_plays"] == 3

    def test_m02_has_zero_completion(self, store):
        # m02: 1 session, not completed -> 0 %
        rates = store.completion_rate_by_movie(["m02"])
        assert rates[0]["completion_pct"] == 0.0

    def test_m03_has_partial_completion(self, store):
        # m03: 3 sessions, 2 completed -> 66.7 %
        rates = store.completion_rate_by_movie(["m03"])
        assert abs(rates[0]["completion_pct"] - 66.7) < 0.2

    def test_sorted_by_completion_pct_desc(self, store):
        rates = store.completion_rate_by_movie(["m01", "m02", "m03"])
        pcts = [r["completion_pct"] for r in rates]
        assert pcts == sorted(pcts, reverse=True)

    def test_all_movies_returned_when_no_filter(self, store):
        rates = store.completion_rate_by_movie()
        movie_ids = {r["movie_id"] for r in rates}
        assert {"m01", "m02", "m03"}.issubset(movie_ids)


# -- Cohort retention

class TestCohortRetention:

    def test_single_cohort_on_first_day(self, store):
        cohorts = store.cohort_retention(_FULL_START, _FULL_END)
        assert len(cohorts) == 1
        assert cohorts[0]["cohort_day"] == "2024-03-01"

    def test_cohort_size_equals_two(self, store):
        cohorts = store.cohort_retention(_FULL_START, _FULL_END)
        assert cohorts[0]["cohort_size"] == 2

    def test_day_zero_retention_equals_cohort_size(self, store):
        cohorts = store.cohort_retention(_FULL_START, _FULL_END)
        row = cohorts[0]
        assert row["day_0"] == row["cohort_size"]

    def test_day_one_retention(self, store):
        # u01 watched on 2024-03-02 -> day_1 = 1
        cohorts = store.cohort_retention(_FULL_START, _FULL_END)
        assert cohorts[0]["day_1"] == 1

    def test_day_seven_retention(self, store):
        # u02 watched on 2024-03-08 (= cohort_day + 7) -> day_7 = 1
        cohorts = store.cohort_retention(_FULL_START, _FULL_END)
        assert cohorts[0]["day_7"] == 1

    def test_empty_store_returns_no_cohorts(self):
        assert EngagementStore().cohort_retention(_FULL_START, _FULL_END) == []


# -- Event type breakdown

class TestEventTypeBreakdown:

    def test_play_is_most_common_event(self, store):
        breakdown = store.event_type_breakdown(_FULL_START, _FULL_END)
        assert breakdown[0]["event_type"] == "play"
        assert breakdown[0]["event_count"] == 4

    def test_sorted_descending_by_event_count(self, store):
        breakdown = store.event_type_breakdown(_FULL_START, _FULL_END)
        counts = [r["event_count"] for r in breakdown]
        assert counts == sorted(counts, reverse=True)

    def test_all_event_types_present(self, store):
        breakdown = store.event_type_breakdown(_FULL_START, _FULL_END)
        types = {r["event_type"] for r in breakdown}
        assert {"play", "pause", "search", "add_to_list"}.issubset(types)

    def test_total_event_count_matches_store(self, store):
        breakdown = store.event_type_breakdown(_FULL_START, _FULL_END)
        assert sum(r["event_count"] for r in breakdown) == store.event_count()

    def test_narrow_window_excludes_later_events(self, store):
        # Only events on 2024-03-01 (5 events: play, pause, play, play, search)
        bd = store.event_type_breakdown(
            "2024-03-01 00:00:00", "2024-03-01 23:59:59"
        )
        total = sum(r["event_count"] for r in bd)
        assert total == 5
