"""Clean text and merge the job listings into one job corpus.

Run:  python -m src.data.preprocess

Outputs:
    data/interim/job_corpus.csv      merged job table (common schema)
    data/processed/jobs_clean.csv    + a cleaned `text` column for TF-IDF
    data/processed/resumes_clean.csv cleaned resume text + Category
"""

import re

import pandas as pd

from src import config
from src.data import load_data


# ----------------------------------------------------------------------------
# Text cleaning
# ----------------------------------------------------------------------------
def clean_text(text):
    """Lowercase, strip URLs/punctuation/extra spaces. Safe on NaN/non-strings."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # drop URLs
    text = re.sub(r"[^a-z0-9+#. ]", " ", text)       # keep words, digits, c++/c#/.net
    text = re.sub(r"\s+", " ", text).strip()          # collapse whitespace
    return text


# ----------------------------------------------------------------------------
# Map each source to the common schema
# ----------------------------------------------------------------------------
def _map_naukri(df):
    """Naukri columns -> title, company, location, skills, description, experience."""
    out = df.rename(columns={
        "jobtitle": "title",
        "company": "company",
        "joblocation_address": "location",
        "skills": "skills",
        "jobdescription": "description",
        "experience": "experience",
    })
    out["source"] = "naukri"
    return out[config.COMMON_JOB_COLUMNS + ["source"]]


def _map_linkedin(df):
    """LinkedIn columns -> the same common schema."""
    out = df.rename(columns={
        "title": "title",
        "company_name": "company",
        "location": "location",
        "skills": "skills",
        "description": "description",
        "formatted_experience_level": "experience",
    })
    out["source"] = "linkedin"
    return out[config.COMMON_JOB_COLUMNS + ["source"]]


# ----------------------------------------------------------------------------
# Build the merged job corpus
# ----------------------------------------------------------------------------
def build_job_corpus():
    """Merge all available job sources into one clean corpus and save it."""
    frames = [_map_naukri(load_data.load_naukri())]

    if load_data.linkedin_available():
        print("LinkedIn found -> merging it into the corpus.")
        frames.append(_map_linkedin(load_data.load_linkedin()))
    else:
        print("LinkedIn not found -> building the corpus from Naukri only.")

    corpus = pd.concat(frames, ignore_index=True)

    # drop rows with no title and no description
    for col in config.COMMON_JOB_COLUMNS:
        corpus[col] = corpus[col].fillna("").astype(str)
    corpus = corpus[(corpus["title"].str.strip() != "") |
                    (corpus["description"].str.strip() != "")]

    # drop duplicate postings
    corpus = corpus.drop_duplicates(subset=["title", "company", "location"])
    corpus = corpus.reset_index(drop=True)

    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(config.JOB_CORPUS_CSV, index=False)
    print(f"Saved {len(corpus):,} jobs -> {config.JOB_CORPUS_CSV}")

    # model-ready version with one cleaned text column
    processed = corpus.copy()
    processed["text"] = (
        processed["title"] + " " + processed["skills"] + " " + processed["description"]
    ).map(clean_text)
    processed = processed[processed["text"].str.strip() != ""]

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed.to_csv(config.JOBS_CLEAN_CSV, index=False)
    print(f"Saved {len(processed):,} model-ready jobs -> {config.JOBS_CLEAN_CSV}")
    return processed


# ----------------------------------------------------------------------------
# Build the cleaned resume table
# ----------------------------------------------------------------------------
def build_clean_resumes():
    """Clean the resume text and save Category + cleaned text."""
    df = load_data.load_resumes()
    df["text"] = df["Resume"].map(clean_text)
    df = df[["Category", "text"]]
    df = df[df["text"].str.strip() != ""]

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RESUMES_CLEAN_CSV, index=False)
    print(f"Saved {len(df):,} resumes -> {config.RESUMES_CLEAN_CSV}")
    return df


def main():
    print("=== Building resume table ===")
    build_clean_resumes()
    print("\n=== Building job corpus ===")
    build_job_corpus()
    print("\nDone.")


if __name__ == "__main__":
    main()