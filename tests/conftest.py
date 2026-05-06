"""Shared fixtures for all tests."""

import sys
import os
import pytest

# Ensure src is on the path when running pytest from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from databaseai.seed_data import MOVIES, USERS, RATINGS
from databaseai.vector_db import MovieVectorStore
from databaseai.relational_db import MovieRegistry
from databaseai.nosql_db import UserSessionStore
from databaseai.feature_store import FeatureStore
from databaseai.rag_pipeline import RAGIngestion, RAGRetrieval


@pytest.fixture
def movies():
    return MOVIES


@pytest.fixture
def users():
    return USERS


@pytest.fixture
def ratings():
    return RATINGS


@pytest.fixture
def vector_store():
    store = MovieVectorStore()  # ephemeral (in-memory)
    store.upsert_movies(MOVIES)
    return store


@pytest.fixture
def registry():
    reg = MovieRegistry()       # SQLite in-memory
    reg.add_movies(MOVIES)
    for u in USERS:
        reg.add_user(u)
    for user_id, movie_id, score, review in RATINGS:
        reg.add_rating(user_id, movie_id, score, review)
    return reg


@pytest.fixture
def session_store():
    return UserSessionStore()   # in-memory


@pytest.fixture
def feature_store(registry):
    fs = FeatureStore()
    for u in USERS:
        user_ratings = registry.get_user_ratings(u["id"])
        fs.compute_user_features(
            user_id=u["id"],
            ratings=user_ratings,
            watch_count_7d=5,
            watch_count_30d=15,
        )
    for m in MOVIES:
        stats = registry.movie_stats(m["id"])
        ratings_list = []
        if stats["total_ratings"] and stats["avg_score"]:
            ratings_list = [stats["avg_score"]] * int(stats["total_ratings"])
        fs.compute_movie_features(m["id"], m["genre"], ratings_list)
    return fs


@pytest.fixture
def rag(tmp_path):
    ingestion = RAGIngestion()
    docs = [
        {
            "id": m["id"],
            "title": m["title"],
            "content": m["description"],
            "metadata": {"genre": m["genre"], "year": m["year"]},
        }
        for m in MOVIES
    ]
    ingestion.ingest_batch(docs)
    retrieval = RAGRetrieval.from_ingestion(ingestion)
    return ingestion, retrieval
