"""Tests for the Feature Store layer."""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from databaseai.feature_store import FeatureStore


class TestFeatureStore:

    def test_user_feature_count(self, feature_store):
        counts = feature_store.feature_count()
        assert counts["user_features"] == 5  # 5 users

    def test_movie_feature_count(self, feature_store):
        counts = feature_store.feature_count()
        assert counts["movie_features"] == 20  # 20 movies

    def test_get_user_features(self, feature_store):
        feats = feature_store.get_user_features("u01")
        assert feats is not None
        assert feats["avg_rating"] > 0
        assert feats["total_ratings"] > 0
        assert feats["fav_genre"] in ["sci-fi", "action", "drama", "thriller", "horror", "animation", "romance", "unknown"]

    def test_user_u01_prefers_scifi(self, feature_store):
        # u01 rated m01 (sci-fi), m03 (sci-fi), m04 (sci-fi), m11 (sci-fi)
        feats = feature_store.get_user_features("u01")
        assert feats["fav_genre"] == "sci-fi"

    def test_get_movie_features(self, feature_store):
        feats = feature_store.get_movie_features("m01")
        assert feats is not None
        assert isinstance(feats["popularity_score"], float)
        assert isinstance(feats["genre_vector"], dict)
        assert feats["genre_vector"].get("sci-fi") == 1.0

    def test_popularity_score_range(self, feature_store):
        for movie_id in ["m01", "m07", "m15"]:
            feats = feature_store.get_movie_features(movie_id)
            assert 0.0 <= feats["popularity_score"] <= 10.0

    def test_genre_vector_sums_to_one(self, feature_store):
        feats = feature_store.get_movie_features("m01")
        total = sum(feats["genre_vector"].values())
        assert abs(total - 1.0) < 1e-6

    def test_top_movies_by_popularity(self, feature_store):
        top = feature_store.get_top_movies_by_popularity(n=5)
        assert len(top) <= 5
        if len(top) > 1:
            scores = [r["popularity_score"] for r in top]
            assert scores == sorted(scores, reverse=True)

    def test_nonexistent_user_returns_none(self, feature_store):
        assert feature_store.get_user_features("u99") is None

    def test_nonexistent_movie_returns_none(self, feature_store):
        assert feature_store.get_movie_features("m99") is None

    def test_compute_user_features_no_ratings(self):
        fs = FeatureStore()
        feats = fs.compute_user_features("u_new", ratings=[])
        assert feats["avg_rating"] == 0.0
        assert feats["fav_genre"] == "unknown"
        assert feats["total_ratings"] == 0

    def test_point_in_time_snapshot(self, feature_store):
        snapshot = feature_store.get_training_snapshot("2099-01-01T00:00:00+00:00")
        assert len(snapshot) == 5

    def test_feature_versioning(self):
        fs_v1 = FeatureStore(db_path=":memory:", version="v1")
        fs_v2 = FeatureStore(db_path=":memory:", version="v2")

        fs_v1.compute_user_features("u01", [{"score": 4.0, "genre": "drama"}])
        fs_v2.compute_user_features("u01", [{"score": 5.0, "genre": "sci-fi"}])

        f1 = fs_v1.get_user_features("u01")
        f2 = fs_v2.get_user_features("u01")
        assert f1["avg_rating"] == 4.0
        assert f2["avg_rating"] == 5.0
