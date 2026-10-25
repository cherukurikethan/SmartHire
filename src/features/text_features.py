
"""Convert cleaned job and resume text into TF-IDF features."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config


def build_tfidf():
    """Build TF-IDF features from cleaned job and resume text."""

    jobs = pd.read_csv(config.JOBS_CLEAN_CSV)
    resumes = pd.read_csv(config.RESUMES_CLEAN_CSV)

    job_text = jobs["text"].fillna("").astype(str)
    resume_text = resumes["text"].fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    all_text = pd.concat(
        [job_text, resume_text],
        ignore_index=True
    )

    tfidf_matrix = vectorizer.fit_transform(all_text)

    model_dir = Path(config.MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "tfidf_vectorizer.joblib"
    joblib.dump(vectorizer, model_path)

    print("TF-IDF feature extraction successful!")
    print("Jobs:", len(job_text))
    print("Resumes:", len(resume_text))
    print("Total text records:", len(all_text))
    print("TF-IDF matrix shape:", tfidf_matrix.shape)
    print("Vocabulary size:", len(vectorizer.vocabulary_))
    print("Saved vectorizer to:", model_path)

    return tfidf_matrix, vectorizer


if __name__ == "__main__":
    build_tfidf()
