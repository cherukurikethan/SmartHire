
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "naukri_com-job_sample.csv"


def match_jobs(resume_text, top_n=5):
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Find available job description columns
    possible_columns = [
        "jobdescription",
        "job_description",
        "Job Description",
        "description",
    ]

    description_col = next(
        (col for col in possible_columns if col in df.columns),
        None,
    )

    if description_col is None:
        raise ValueError(
            f"Job description column not found. Available columns: {list(df.columns)}"
        )

    # Remove jobs without descriptions
    df = df.dropna(subset=[description_col]).copy()
    df[description_col] = df[description_col].astype(str)

    # Compare resume with job descriptions
    texts = [resume_text] + df[description_col].tolist()

    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    matrix = vectorizer.fit_transform(texts)

    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    df["match_percentage"] = (scores * 100).round(2)

    # Show the best matches
    top_jobs = df.nlargest(top_n, "match_percentage")

    display_columns = [
        col for col in [
            "match_percentage",
            "Job Title",
            "jobtitle",
            "Title",
            "Company",
            "company",
            description_col,
        ]
        if col in top_jobs.columns
    ]

    return top_jobs[display_columns]


if __name__ == "__main__":
    resume = input("Paste your resume text here: ")

    try:
        results = match_jobs(resume)

        print("\nTop matching jobs:")
        print(results.to_string(index=False))

    except Exception as error:
        print("\nError:", error)
