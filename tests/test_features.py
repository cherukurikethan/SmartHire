
"""Tests for SmartHire text feature extraction and matching."""

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def test_tfidf_feature_extraction():
    """TF-IDF should create features for multiple documents."""
    documents = [
        "Python machine learning data science",
        "Java web development",
        "Python data analysis",
    ]

    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(documents)

    assert features.shape[0] == 3
    assert features.shape[1] > 0


def test_similar_text_has_positive_similarity():
    """Similar texts should have a positive similarity score."""
    documents = [
        "Python machine learning data science",
        "Python machine learning",
        "Java web development",
    ]

    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(documents)

    scores = cosine_similarity(features[0:1], features[1:]).flatten()

    assert scores[0] > 0
    assert scores[1] == pytest.approx(0.0)


def test_transform_new_resume_text():
    """The fitted vectorizer should transform new resume text."""
    documents = [
        "Python machine learning",
        "Java web development",
    ]

    vectorizer = TfidfVectorizer()
    vectorizer.fit(documents)

    resume_vector = vectorizer.transform(
        ["Python machine learning experience"]
    )

    assert resume_vector.shape[0] == 1
    assert resume_vector.shape[1] > 0
    assert resume_vector.nnz > 0


def test_empty_text_produces_zero_similarity():
    """An empty query should produce zero similarity."""
    documents = [
        "Python machine learning",
        "Java web development",
    ]

    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(documents)

    empty_vector = vectorizer.transform([""])
    scores = cosine_similarity(empty_vector, features).flatten()

    assert all(score == pytest.approx(0.0) for score in scores)
