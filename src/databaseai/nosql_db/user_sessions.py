"""
NoSQL / Document Database — JSON File Store
============================================

Stores user watch sessions and interaction events as schema-free documents.

DB Architect notes:
  - Production equivalent: MongoDB, DynamoDB, Firestore, CouchDB
  - Documents are JSON objects — no fixed schema, nested objects fine
  - No joins: denormalize data into the document for read performance
  - Partitioned by user_id (= MongoDB shard key / DynamoDB partition key)
  - Eventual consistency: reads may lag writes slightly (not modelled here)
  - TTL indexes: MongoDB and DynamoDB can auto-expire old sessions
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional


class UserSessionStore:
    """
    Document store for user watch sessions and events.

    Document shape (flexible — new event types just add new keys):
    {
        "session_id": "sess_abc123",
        "user_id": "u42",
        "started_at": "2024-01-15T10:30:00Z",
        "ended_at": null,
        "device": {"type": "smart_tv", "model": "Samsung QN90B"},
        "events": [
            {"type": "search",  "query": "sci-fi", "ts": "..."},
            {"type": "play",    "movie_id": "m101", "ts": "..."},
            {"type": "pause",   "movie_id": "m101", "position_s": 1823, "ts": "..."},
            {"type": "resume",  "movie_id": "m101", "position_s": 1823, "ts": "..."},
            {"type": "rate",    "movie_id": "m101", "score": 5, "ts": "..."}
        ]
    }

    Real-world parallel:
      Netflix logs every user interaction (play, pause, seek, search, exit)
      as a stream of events stored in a document store / event log.
      This data feeds recommendation models and A/B test analysis.
    """

    def __init__(self, store_path: Optional[str] = None):
        self._path = store_path
        # In-memory store: { user_id: [session_doc, ...] }
        self._store: dict[str, list[dict]] = {}
        if store_path and os.path.exists(store_path):
            self._load()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(self, user_id: str, device: dict) -> str:
        session_id = f"sess_{user_id}_{self._now_ts()}"
        doc = {
            "session_id": session_id,
            "user_id": user_id,
            "started_at": self._now(),
            "ended_at": None,
            "device": device,
            "events": [],
        }
        self._store.setdefault(user_id, []).append(doc)
        self._save()
        return session_id

    def end_session(self, user_id: str, session_id: str) -> bool:
        session = self._get_session(user_id, session_id)
        if not session:
            return False
        session["ended_at"] = self._now()
        self._save()
        return True

    def add_event(self, user_id: str, session_id: str, event: dict) -> bool:
        """Append any event to the session — schema is open."""
        session = self._get_session(user_id, session_id)
        if not session:
            return False
        session["events"].append({"ts": self._now(), **event})
        self._save()
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_session(self, user_id: str, session_id: str) -> Optional[dict]:
        return self._get_session(user_id, session_id)

    def get_user_sessions(self, user_id: str, limit: int = 10) -> list[dict]:
        sessions = self._store.get(user_id, [])
        return sessions[-limit:]

    def get_watch_history(self, user_id: str) -> list[str]:
        """Return list of movie_ids the user has started watching."""
        played: list[str] = []
        for session in self._store.get(user_id, []):
            for event in session["events"]:
                if event.get("type") == "play" and event.get("movie_id"):
                    mid = event["movie_id"]
                    if mid not in played:
                        played.append(mid)
        return played

    def get_event_counts(self, user_id: str) -> dict[str, int]:
        """Aggregate event type counts across all sessions."""
        counts: dict[str, int] = {}
        for session in self._store.get(user_id, []):
            for event in session["events"]:
                etype = event.get("type", "unknown")
                counts[etype] = counts.get(etype, 0) + 1
        return counts

    def session_count(self, user_id: Optional[str] = None) -> int:
        if user_id:
            return len(self._store.get(user_id, []))
        return sum(len(v) for v in self._store.values())

    def all_user_ids(self) -> list[str]:
        return list(self._store.keys())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_session(self, user_id: str, session_id: str) -> Optional[dict]:
        for s in self._store.get(user_id, []):
            if s["session_id"] == session_id:
                return s
        return None

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _now_ts() -> str:
        return str(int(datetime.now(timezone.utc).timestamp()))

    def _save(self) -> None:
        if self._path:
            with open(self._path, "w") as f:
                json.dump(self._store, f, indent=2)

    def _load(self) -> None:
        with open(self._path) as f:
            self._store = json.load(f)
