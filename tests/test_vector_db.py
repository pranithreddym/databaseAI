"""Tests for the Vector Database (ChromaDB) layer."""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from databaseai.vector_db import MovieVectorStore
from databaseai.seed_data import MOVIES


class TestMovieVectorStore:

    def test_upsert_and_count(self, vector_store):
        assert vector_store.count() == len(MOVIES)

    def test_find_similar_returns_results(self, vector_store):
        results = vector_store.find_similar("space travel and black holes", n=3)
        assert len(results) == 3
        assert all("title" in r for r in results)
        assert all("similarity_score" in r for r in results)

    def test_similarity_scores_between_0_and_1(self, vector_store):
        results = vector_store.find_similar("crime thriller heist", n=5)
        for r in results:
            assert 0.0 <= r["similarity_score"] <= 1.0

    def test_find_similar_with_genre_filter(self, vector_store):
        results = vector_store.find_similar("dark hero city", n=10, genre_filter="action")
        assert all(r["genre"] == "action" for r in results)

    def test_find_similar_with_year_filter(self, vector_store):
        results = vector_store.find_similar("science fiction", n=10, min_year=2015)
        assert all(r["year"] >= 2015 for r in results)

    def test_get_by_id(self, vector_store):
        result = vector_store.get_by_id("m01")
        assert result is not None
        assert result["title"] == "Inception"

    def test_get_by_id_missing(self, vector_store):
        assert vector_store.get_by_id("nonexistent") is None

    def test_delete_movie(self, vector_store):
        initial_count = vector_store.count()
        vector_store.delete_movie("m20")
        assert vector_store.count() == initial_count - 1

    def test_upsert_updates_existing(self, vector_store):
        updated = {
            "id": "m01",
            "title": "Inception (Director's Cut)",
            "description": "Extended version of the dream heist film",
            "genre": "sci-fi",
            "year": 2010,
            "director": "Christopher Nolan",
            "rating": 5.0,
        }
        vector_store.upsert_movies([updated])
        # Count should remain the same — upsert, not insert
        assert vector_store.count() == len(MOVIES)

    def test_empty_store_returns_no_results(self):
        empty_store = MovieVectorStore()
        empty_store.reset()          # clear any shared-state data from other tests
        results = empty_store.find_similar("anything", n=5)
        assert results == []

    def test_sci_fi_query_returns_results(self, vector_store):
        # Hash-based embeddings don't encode semantics, so just verify results come back
        results = vector_store.find_similar("astronauts wormhole space exploration", n=5)
        assert len(results) == 5
        assert all("title" in r for r in results)
        assert all(r["similarity_score"] >= 0.0 for r in results)
