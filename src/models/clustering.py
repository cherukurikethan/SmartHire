
"""K-Means job clustering and NMF topic modeling."""

import joblib
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config


def run_clustering():
    # Load cleaned job data
    jobs = pd.read_csv(config.JOBS_CLEAN_CSV)
    jobs["text"] = jobs["text"].fillna("").astype(str)

    # Build TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=3000,
        stop_words="english"
    )
    X = vectorizer.fit_transform(jobs["text"])

    # K-Means clustering
    kmeans = MiniBatchKMeans(
        n_clusters=8,
        random_state=42,
        batch_size=2048,
        n_init=3
    )
    jobs["cluster"] = kmeans.fit_predict(X)

    # Save clustered jobs
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.PROCESSED_DIR / "jobs_clustered.csv"
    jobs.to_csv(output_path, index=False)

    # NMF topic modeling
    nmf = NMF(
        n_components=8,
        random_state=42,
        init="nndsvda",
        max_iter=200
    )
    topic_matrix = nmf.fit_transform(X)
    jobs["topic"] = topic_matrix.argmax(axis=1)

    # Save jobs with topic labels
    topic_path = config.PROCESSED_DIR / "jobs_clustered_topics.csv"
    jobs.to_csv(topic_path, index=False)

    # Save models and vectorizer
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans, config.MODEL_DIR / "job_kmeans.joblib")
    joblib.dump(nmf, config.MODEL_DIR / "job_nmf.joblib")
    joblib.dump(vectorizer, config.MODEL_DIR / "clustering_vectorizer.joblib")

    print("Job clustering and NMF topic modeling completed!")
    print("Total jobs:", len(jobs))
    print("\nJobs per cluster:")
    print(jobs["cluster"].value_counts().sort_index())
    print("\nJobs per topic:")
    print(jobs["topic"].value_counts().sort_index())
    print("Clustered jobs saved to:", output_path)
    print("Topic-labeled jobs saved to:", topic_path)


if __name__ == "__main__":
    run_clustering()
