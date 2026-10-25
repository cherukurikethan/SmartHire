
"""Recommend jobs that best match a resume."""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src import config


def recommend_jobs(resume_text, top_n=5):
    """Return the top matching jobs for the given resume text."""

    import joblib

    jobs = pd.read_csv(config.JOBS_CLEAN_CSV)
    jobs["text"] = jobs["text"].fillna("").astype(str)

    vectorizer_path = config.MODEL_DIR / "tfidf_vectorizer.joblib"
    vectorizer = joblib.load(vectorizer_path)

    job_vectors = vectorizer.transform(jobs["text"])
    resume_vector = vectorizer.transform([resume_text])

    scores = cosine_similarity(resume_vector, job_vectors).flatten()
    top_indices = scores.argsort()[::-1][:top_n]

    results = jobs.iloc[top_indices].copy()
    results["similarity_score"] = scores[top_indices].round(4)

    columns = [
        col for col in [
            "title", "company", "location",
            "skills", "similarity_score"
        ]
        if col in results.columns
    ]

    return results[columns].reset_index(drop=True)


if __name__ == "__main__":
    sample_resume = (
        "Python machine learning data science pandas "
        "SQL statistics and data analysis"
    )

    recommendations = recommend_jobs(sample_resume)

    print("Top matching jobs:")
    print(recommendations.to_string(index=False))
