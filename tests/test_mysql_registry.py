"""
Tests for the MySQL relational database backend.

These tests require a running MySQL server.
They are automatically skipped when no server is reachable.

Start a local server with Docker:
  docker run -d --name mysql-cineai \\
    -e MYSQL_ROOT_PASSWORD=cineai \\
    -e MYSQL_DATABASE=cineai \\
    -p 3306:3306 mysql:8
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pymysql
from databaseai.relational_db import MySQLMovieRegistry
from databaseai.seed_data import MOVIES, USERS, RATINGS

MYSQL_CONFIG = dict(
    host="127.0.0.1",
    port=3306,
    user="root",
    password="cineai",
    database="cineai",
)


def mysql_available() -> bool:
    try:
        conn = pymysql.connect(**MYSQL_CONFIG, connect_timeout=2)
        conn.close()
        return True
    except Exception:
        return False


requires_mysql = pytest.mark.skipif(
    not mysql_available(),
    reason="MySQL server not reachable at 127.0.0.1:3306 — start with: "
           "docker run -d --name mysql-cineai -e MYSQL_ROOT_PASSWORD=cineai "
           "-e MYSQL_DATABASE=cineai -p 3306:3306 mysql:8",
)


@pytest.fixture
def mysql_registry():
    reg = MySQLMovieRegistry(**MYSQL_CONFIG)
    reg.teardown()          # clean slate before each test
    reg._init_schema()      # recreate tables
    reg.add_movies(MOVIES)
    for u in USERS:
        reg.add_user(u)
    for user_id, movie_id, score, review in RATINGS:
        reg.add_rating(user_id, movie_id, score, review)
    yield reg
    reg.teardown()          # clean up after


@requires_mysql
class TestMySQLMovieRegistry:

    def test_movie_count(self, mysql_registry):
        assert mysql_registry.movie_count() == len(MOVIES)

    def test_user_count(self, mysql_registry):
        assert mysql_registry.user_count() == len(USERS)

    def test_rating_count(self, mysql_registry):
        assert mysql_registry.rating_count() == len(RATINGS)

    def test_get_movie(self, mysql_registry):
        movie = mysql_registry.get_movie("m01")
        assert movie is not None
        assert movie["title"] == "Inception"
        assert movie["genre"] == "sci-fi"
        assert movie["year"] == 2010

    def test_get_nonexistent_movie(self, mysql_registry):
        assert mysql_registry.get_movie("m99") is None

    def test_search_by_genre(self, mysql_registry):
        results = mysql_registry.search_movies(genre="sci-fi")
        assert len(results) > 0
        assert all(r["genre"] == "sci-fi" for r in results)

    def test_search_by_year(self, mysql_registry):
        results = mysql_registry.search_movies(min_year=2019)
        assert all(r["year"] >= 2019 for r in results)

    def test_search_by_director(self, mysql_registry):
        results = mysql_registry.search_movies(director="Nolan")
        assert len(results) >= 3
        assert all("Nolan" in r["director"] for r in results)

    def test_top_rated_movies(self, mysql_registry):
        top = mysql_registry.top_rated_movies(n=5)
        assert len(top) <= 5
        if len(top) > 1:
            scores = [float(r["avg_score"]) for r in top]
            assert scores == sorted(scores, reverse=True)

    def test_top_rated_by_genre(self, mysql_registry):
        top = mysql_registry.top_rated_movies(genre="sci-fi", n=5)
        assert all(r["genre"] == "sci-fi" for r in top)

    def test_add_rating_upsert(self, mysql_registry):
        mysql_registry.add_rating("u01", "m02", 4.0, "First watch")
        mysql_registry.add_rating("u01", "m02", 5.0, "Re-watched, even better")
        ratings = mysql_registry.get_user_ratings("u01")
        m02_ratings = [r for r in ratings if r["title"] == "The Dark Knight"]
        assert len(m02_ratings) == 1
        assert m02_ratings[0]["score"] == 5.0

    def test_foreign_key_constraint(self, mysql_registry):
        with pytest.raises(Exception):
            mysql_registry.add_rating("nonexistent_user", "m01", 5.0)

    def test_get_user_ratings(self, mysql_registry):
        ratings = mysql_registry.get_user_ratings("u01")
        assert len(ratings) > 0
        assert all("title" in r for r in ratings)
        assert all("score" in r for r in ratings)

    def test_movie_stats(self, mysql_registry):
        stats = mysql_registry.movie_stats("m01")
        assert stats["total_ratings"] >= 3
        assert 1.0 <= float(stats["avg_score"]) <= 5.0

    def test_add_duplicate_user_ignored(self, mysql_registry):
        initial = mysql_registry.user_count()
        mysql_registry.add_user(USERS[0])
        assert mysql_registry.user_count() == initial

    def test_on_duplicate_key_update_movie(self, mysql_registry):
        updated = {**MOVIES[0], "title": "Inception (Director's Cut)", "description": "Extended cut"}
        mysql_registry.add_movie(updated)
        movie = mysql_registry.get_movie("m01")
        assert movie["title"] == "Inception (Director's Cut)"
        assert mysql_registry.movie_count() == len(MOVIES)
