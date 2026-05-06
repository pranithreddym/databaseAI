"""Tests for the Relational Database (SQLite) layer."""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from databaseai.relational_db import MovieRegistry
from databaseai.seed_data import MOVIES, USERS, RATINGS


class TestMovieRegistry:

    def test_movie_count(self, registry):
        assert registry.movie_count() == len(MOVIES)

    def test_user_count(self, registry):
        assert registry.user_count() == len(USERS)

    def test_rating_count(self, registry):
        assert registry.rating_count() == len(RATINGS)

    def test_get_movie(self, registry):
        movie = registry.get_movie("m01")
        assert movie is not None
        assert movie["title"] == "Inception"
        assert movie["genre"] == "sci-fi"
        assert movie["year"] == 2010

    def test_get_nonexistent_movie(self, registry):
        assert registry.get_movie("m99") is None

    def test_search_by_genre(self, registry):
        results = registry.search_movies(genre="sci-fi")
        assert len(results) > 0
        assert all(r["genre"] == "sci-fi" for r in results)

    def test_search_by_year(self, registry):
        results = registry.search_movies(min_year=2019)
        assert all(r["year"] >= 2019 for r in results)

    def test_search_by_director(self, registry):
        results = registry.search_movies(director="Nolan")
        assert len(results) >= 3
        assert all("Nolan" in r["director"] for r in results)

    def test_top_rated_movies(self, registry):
        top = registry.top_rated_movies(n=5)
        assert len(top) <= 5
        if len(top) > 1:
            scores = [r["avg_score"] for r in top]
            assert scores == sorted(scores, reverse=True)

    def test_top_rated_by_genre(self, registry):
        top = registry.top_rated_movies(genre="sci-fi", n=5)
        assert all("sci-fi" == r["genre"] for r in top)

    def test_add_rating_upsert(self, registry):
        # Rate, then re-rate — should upsert, not duplicate
        registry.add_rating("u01", "m02", 4.0, "First watch")
        registry.add_rating("u01", "m02", 5.0, "Re-watched, even better")
        ratings = registry.get_user_ratings("u01")
        m02_ratings = [r for r in ratings if r["title"] == "The Dark Knight"]
        assert len(m02_ratings) == 1
        assert m02_ratings[0]["score"] == 5.0

    def test_foreign_key_constraint(self, registry):
        with pytest.raises(Exception):
            registry.add_rating("nonexistent_user", "m01", 5.0)

    def test_get_user_ratings(self, registry):
        ratings = registry.get_user_ratings("u01")
        assert len(ratings) > 0
        assert all("title" in r for r in ratings)
        assert all("score" in r for r in ratings)

    def test_movie_stats(self, registry):
        stats = registry.movie_stats("m01")
        assert stats["total_ratings"] >= 3
        assert 1.0 <= stats["avg_score"] <= 5.0

    def test_add_duplicate_user_ignored(self, registry):
        initial = registry.user_count()
        registry.add_user(USERS[0])  # already exists
        assert registry.user_count() == initial
