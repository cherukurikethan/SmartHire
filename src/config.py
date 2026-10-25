
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directory
MODEL_DIR = ROOT_DIR / "models"

# Resume dataset
RESUME_DATASET = RAW_DATA_DIR / "UpdatedResumeDataSet.csv"

# Job dataset
JOB_DATASET = RAW_DATA_DIR / "naukri_com-job_sample.csv"

# Saved classifier
CLASSIFIER_MODEL = MODEL_DIR / "resume_classifier.joblib"
 

RESUME_DATASET = RAW_DATA_DIR / "UpdatedResumeDataSet.csv"
JOB_DATASET = RAW_DATA_DIR / "naukri_com-job_sample.csv"

RESUME_CSV = RESUME_DATASET
NAUKRI_CSV = JOB_DATASET

LINKEDIN_POSTINGS_CSV = RAW_DATA_DIR / "linkedin" / "postings.csv"
LINKEDIN_JOB_SKILLS_CSV = RAW_DATA_DIR / "linkedin" / "jobs" / "job_skills.csv"
LINKEDIN_SKILLS_MAP_CSV = RAW_DATA_DIR / "linkedin" / "mappings" / "skills.csv"

INTERIM_DIR = INTERIM_DATA_DIR
PROCESSED_DIR = PROCESSED_DATA_DIR

RESUMES_CLEAN_CSV = PROCESSED_DIR / "resumes_clean.csv"

COMMON_JOB_COLUMNS = [
    "title",
    "company",
    "location",
    "skills",
    "description",
    "experience",
]

JOB_CORPUS_CSV = INTERIM_DIR / "job_corpus.csv"
JOBS_CLEAN_CSV = PROCESSED_DIR / "jobs_clean.csv"