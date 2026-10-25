
"""Estimate resume-job fit using TF-IDF similarity."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src import config


def predict_fit(resume_text, top_n=5):
    """Rank jobs by similarity to the supplied resume text."""

    jobs = pd.read_csv(config.JOBS_CLEAN_CSV)
    jobs["text"] = jobs["text"].fillna("").astype(str)

    if not isinstance(resume_text, str) or not resume_text.strip():
        raise ValueError("Please provide non-empty resume text.")

    # Fit the vectorizer on the job corpus and transform the resume
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    job_vectors = vectorizer.fit_transform(jobs["text"])
    resume_vector = vectorizer.transform([resume_text])

    scores = cosine_similarity(
        resume_vector, job_vectors
    ).flatten()

    top_indices = scores.argsort()[::-1][:top_n]
    results = jobs.iloc[top_indices].copy()
    results["fit_score"] = (scores[top_indices] * 100).round(2)

    columns = [
        col for col in [
            "title", "company", "location", "skills", "fit_score"
        ]
        if col in results.columns
    ]

    return results[columns].reset_index(drop=True)


if __name__ == "__main__":
    sample_resume = """
    Python, machine learning, data science, SQL,
    pandas, data analysis and statistics
    """

    recommendations = predict_fit(sample_resume)

    print("Resume-job fit prediction completed!")
    print(recommendations.to_string(index=False))

