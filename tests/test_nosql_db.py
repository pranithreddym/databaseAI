"""Tests for the NoSQL / Document Store layer."""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from databaseai.nosql_db import UserSessionStore


class TestUserSessionStore:

    def test_create_session(self, session_store):
        sid = session_store.create_session("u01", {"type": "web"})
        assert sid.startswith("sess_u01_")
        assert session_store.session_count("u01") == 1

    def test_add_and_retrieve_events(self, session_store):
        sid = session_store.create_session("u01", {"type": "mobile"})
        session_store.add_event("u01", sid, {"type": "search", "query": "sci-fi"})
        session_store.add_event("u01", sid, {"type": "play", "movie_id": "m01"})
        session_store.add_event("u01", sid, {"type": "pause", "movie_id": "m01", "position_s": 3600})

        session = session_store.get_session("u01", sid)
        assert session is not None
        assert len(session["events"]) == 3
        assert session["events"][0]["type"] == "search"
        assert session["events"][1]["type"] == "play"
        assert session["events"][2]["position_s"] == 3600

    def test_watch_history(self, session_store):
        sid = session_store.create_session("u02", {"type": "tv"})
        session_store.add_event("u02", sid, {"type": "play", "movie_id": "m02"})
        session_store.add_event("u02", sid, {"type": "play", "movie_id": "m10"})
        session_store.add_event("u02", sid, {"type": "play", "movie_id": "m02"})  # duplicate

        history = session_store.get_watch_history("u02")
        assert "m02" in history
        assert "m10" in history
        assert history.count("m02") == 1  # deduplicated

    def test_event_counts(self, session_store):
        sid = session_store.create_session("u03", {"type": "web"})
        for _ in range(3):
            session_store.add_event("u03", sid, {"type": "search", "query": "action"})
        session_store.add_event("u03", sid, {"type": "play", "movie_id": "m10"})

        counts = session_store.get_event_counts("u03")
        assert counts["search"] == 3
        assert counts["play"] == 1

    def test_end_session(self, session_store):
        sid = session_store.create_session("u04", {"type": "web"})
        assert session_store.get_session("u04", sid)["ended_at"] is None
        session_store.end_session("u04", sid)
        assert session_store.get_session("u04", sid)["ended_at"] is not None

    def test_add_event_to_nonexistent_session(self, session_store):
        result = session_store.add_event("u01", "sess_fake", {"type": "play"})
        assert result is False

    def test_flexible_event_schema(self, session_store):
        """Documents can hold any shape — no schema enforcement."""
        sid = session_store.create_session("u05", {"type": "web"})
        session_store.add_event("u05", sid, {
            "type": "recommendation_click",
            "algorithm": "collaborative_filtering",
            "position": 3,
            "movie_id": "m16",
            "ab_variant": "B",
        })
        session = session_store.get_session("u05", sid)
        event = session["events"][0]
        assert event["algorithm"] == "collaborative_filtering"
        assert event["ab_variant"] == "B"

    def test_multiple_users_isolated(self, session_store):
        s1 = session_store.create_session("u01", {"type": "web"})
        s2 = session_store.create_session("u02", {"type": "tv"})
        session_store.add_event("u01", s1, {"type": "play", "movie_id": "m01"})

        assert session_store.get_watch_history("u01") == ["m01"]
        assert session_store.get_watch_history("u02") == []

    def test_total_session_count(self, session_store):
        session_store.create_session("u01", {"type": "web"})
        session_store.create_session("u01", {"type": "mobile"})
        session_store.create_session("u02", {"type": "tv"})
        assert session_store.session_count() == 3
