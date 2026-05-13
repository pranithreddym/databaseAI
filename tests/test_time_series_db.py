"""Tests for the Time-Series Metrics Store."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from databaseai.time_series_db import MetricsStore


@pytest.fixture
def store():
    s = MetricsStore()
    for h in range(48):
        if h < 24:
            ts = f"2024-01-01 {h:02d}:00:00"
        else:
            ts = f"2024-01-02 {h - 24:02d}:00:00"
        s.record_metric("v1.0", "control",   "precision_at_10", round(0.350 + h * 0.001, 4), ts)
        s.record_metric("v1.1", "treatment", "precision_at_10", round(0.380 + h * 0.001, 4), ts)
        s.record_metric("v1.0", "control",   "ndcg_at_10",      round(0.450 + h * 0.001, 4), ts)
        s.record_metric("v1.1", "treatment", "ndcg_at_10",      round(0.500 + h * 0.001, 4), ts)
    s.record_event("deployment", "v1.0 deployed", {"version": "v1.0"}, "2024-01-01 00:00:00")
    s.record_event("ab_start", "A/B test started", {"treatment_pct": 10}, "2024-01-01 12:00:00")
    return s


class TestMetricsStore:

    def test_metric_count(self, store):
        assert store.metric_count() == 192

    def test_event_count(self, store):
        assert store.event_count() == 2

    def test_record_metric_returns_row_id(self):
        s = MetricsStore()
        row_id = s.record_metric("v1", "control", "ctr", 0.25)
        assert isinstance(row_id, int) and row_id >= 1

    def test_bulk_record_metrics(self):
        s = MetricsStore()
        rows = [("v1", "control", "ctr", 0.20 + i * 0.01, f"2024-06-01 {i:02d}:00:00") for i in range(10)]
        s.bulk_record_metrics(rows)
        assert s.metric_count() == 10

    def test_query_window_correct_range(self, store):
        rows = store.query_window("precision_at_10", "2024-01-01 00:00:00", "2024-01-01 23:00:00")
        assert len(rows) == 48

    def test_query_window_empty_outside_range(self, store):
        rows = store.query_window("precision_at_10", "2030-01-01 00:00:00", "2030-12-31 23:59:59")
        assert rows == []

    def test_query_window_filters_by_variant(self, store):
        rows = store.query_window("precision_at_10", "2024-01-01 00:00:00", "2024-01-01 23:00:00", variant="control")
        assert len(rows) == 24
        assert all(r["variant"] == "control" for r in rows)

    def test_downsample_daily_row_count(self, store):
        daily = store.downsample("precision_at_10", "%Y-%m-%d", "2024-01-01 00:00:00", "2024-01-02 23:00:00")
        assert len(daily) == 4

    def test_downsample_avg_within_min_max(self, store):
        daily = store.downsample("precision_at_10", "%Y-%m-%d", "2024-01-01 00:00:00", "2024-01-01 23:00:00", variant="control")
        assert len(daily) == 1
        row = daily[0]
        assert row["sample_count"] == 24
        assert row["min_value"] <= row["avg_value"] <= row["max_value"]

    def test_downsample_is_less_than_raw(self, store):
        raw   = store.query_window("precision_at_10", "2024-01-01 00:00:00", "2024-01-02 23:00:00")
        daily = store.downsample("precision_at_10", "%Y-%m-%d", "2024-01-01 00:00:00", "2024-01-02 23:00:00")
        assert len(daily) < len(raw)

    def test_ab_test_summary_has_both_variants(self, store):
        summary = store.ab_test_summary("precision_at_10")
        variants = {r["variant"] for r in summary}
        assert "control" in variants and "treatment" in variants

    def test_ab_test_treatment_outperforms_control(self, store):
        by_variant = {r["variant"]: r for r in store.ab_test_summary("precision_at_10")}
        assert by_variant["treatment"]["avg_value"] > by_variant["control"]["avg_value"]

    def test_detect_trend_improving(self):
        s = MetricsStore()
        for i, v in enumerate([0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48]):
            s.record_metric("v1", "control", "prec", v, f"2024-01-01 {i:02d}:00:00")
        assert s.detect_trend("prec", "control", 10)["direction"] == "improving"

    def test_detect_trend_degrading(self):
        s = MetricsStore()
        for i, v in enumerate([0.48, 0.46, 0.44, 0.42, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30]):
            s.record_metric("v1", "control", "prec", v, f"2024-01-01 {i:02d}:00:00")
        assert s.detect_trend("prec", "control", 10)["direction"] == "degrading"

    def test_detect_trend_insufficient_data_is_stable(self):
        s = MetricsStore()
        s.record_metric("v1", "control", "prec", 0.40)
        trend = s.detect_trend("prec", "control", 5)
        assert trend["slope"] == 0.0 and trend["direction"] == "stable"

    def test_latest_metric_returns_most_recent(self, store):
        latest = store.latest_metric("precision_at_10", variant="control")
        assert latest is not None
        assert latest["recorded_at"] >= "2024-01-02"

    def test_latest_metric_missing_returns_none(self):
        assert MetricsStore().latest_metric("nonexistent_metric") is None

    def test_record_event_stores_metadata(self, store):
        early = store.query_events("2024-01-01 00:00:00", "2024-01-01 06:00:00")
        assert len(early) >= 1
        assert isinstance(early[0]["metadata"], dict)
        assert "version" in early[0]["metadata"]

    def test_query_events_respects_time_boundary(self, store):
        before_noon = store.query_events("2024-01-01 00:00:00", "2024-01-01 11:59:59")
        assert len(before_noon) == 1
        assert before_noon[0]["event_type"] == "deployment"
