
import streamlit as st
import pandas as pd
import joblib
import fitz
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="SmartHire",
    page_icon="💼",
    layout="wide"
)

# ---------------- PROJECT PATHS ----------------
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "resume_classifier.joblib"
JOB_PATH = ROOT / "data" / "raw" / "naukri_com-job_sample.csv"

# ---------------- TECHNICAL SKILLS ----------------
TECHNICAL_SKILLS = [
    "Python", "SQL", "Machine Learning", "Deep Learning",
    "Data Analysis", "Pandas", "NumPy", "Scikit-learn",
    "TensorFlow", "PyTorch", "NLP", "Statistics",
    "Excel", "Power BI", "Tableau", "Java", "C++",
    "HTML", "CSS", "JavaScript", "AWS", "Git",
    "Data Visualization", "Data Mining", "Flask",
    "Django", "React", "Linux", "MySQL", "MongoDB",
    "Natural Language Processing", "Artificial Intelligence",
    "Computer Vision", "Big Data", "Hadoop", "Spark"
]


def find_column(df, candidates):
    """Find a column name without case or whitespace differences."""
    lookup = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def extract_skills(text):
    """Find configured technical skills in text."""
    text = str(text).lower()
    found = []

    for skill in TECHNICAL_SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"
        if re.search(pattern, text):
            found.append(skill)

    return found


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    try:
        if MODEL_PATH.exists():
            return joblib.load(MODEL_PATH)
    except Exception as error:
        st.warning(f"Could not load the resume model: {error}")
    return None


# ---------------- LOAD JOB DATA ----------------
@st.cache_data
def load_jobs():
    if not JOB_PATH.exists():
        return pd.DataFrame(), None, None, None

    try:
        df = pd.read_csv(JOB_PATH, low_memory=False)
        df.columns = df.columns.astype(str).str.strip()

        description_col = find_column(
            df,
            [
                "jobdescription", "job_description",
                "Job Description", "description",
                "Job Description "
            ]
        )

        title_col = find_column(
            df, ["jobtitle", "Job Title", "title", "Title"]
        )

        company_col = find_column(
            df, ["company", "Company", "Company Name", "company_name"]
        )

        category_col = find_column(
            df,
            [
                "category", "job category", "job_category",
                "industry", "function", "role"
            ]
        )

        if description_col is None:
            return pd.DataFrame(), None, None, None

        df = df.dropna(subset=[description_col]).copy()
        df["description_text"] = df[description_col].astype(str)

        if title_col is None:
            df["display_title"] = "Job"
        else:
            df["display_title"] = (
                df[title_col].fillna("Job").astype(str)
            )

        if company_col is None:
            df["display_company"] = "Not specified"
        else:
            df["display_company"] = (
                df[company_col].fillna("Not specified").astype(str)
            )

        if category_col is None:
            df["display_category"] = "Not specified"
        else:
            df["display_category"] = (
                df[category_col].fillna("Not specified").astype(str)
            )

        return (
            df.reset_index(drop=True),
            "description_text",
            "display_title",
            "display_category"
        )

    except Exception as error:
        st.warning(f"Could not read the jobs CSV: {error}")
        return pd.DataFrame(), None, None, None


# ---------------- APP HEADER ----------------
st.title("💼 SmartHire")
st.subheader("Resume-to-Job Matching & Career Guidance Engine")
st.write(
    "Upload a resume or paste resume text to predict its category, "
    "find matching jobs, and identify skills to improve."
)
st.divider()

model = load_model()
jobs, description_col, title_col, category_col = load_jobs()

# ---------------- RESUME INPUT ----------------
uploaded_file = st.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"]
)

manual_text = st.text_area(
    "Or paste your resume text",
    height=180,
    placeholder="Python, SQL, Machine Learning, Data Analysis..."
)

resume_text = manual_text.strip()

if uploaded_file is not None:
    try:
        with fitz.open(
            stream=uploaded_file.getvalue(),
            filetype="pdf"
        ) as pdf:
            pdf_text = "\n".join(page.get_text() for page in pdf)

        if pdf_text.strip():
            resume_text = pdf_text
            st.success("Resume PDF uploaded and text extracted!")
            with st.expander("View extracted resume text"):
                st.text(pdf_text)
        else:
            st.warning(
                "No selectable text found in this PDF. "
                "Try a text-based PDF or paste your resume text."
            )
    except Exception as error:
        st.error(f"Could not read the PDF: {error}")


# ---------------- ANALYZE RESUME ----------------
if st.button("Analyze Resume", type="primary"):

    if not resume_text.strip():
        st.warning("Please upload a PDF or paste resume text.")

    else:
        # -------- 1. RESUME CATEGORY --------
        st.header("1. 🎯 Resume Analysis & Job Category")

        if model is not None:
            try:
                predicted_category = model.predict([resume_text])[0]
                st.success(f"Predicted Resume Category: {predicted_category}")
            except Exception as error:
                st.error(f"Resume classification failed: {error}")
        else:
            st.error(
                "Trained model not found. Check "
                "models/resume_classifier.joblib."
            )

        # -------- 2. MATCHING JOBS --------
        st.header("2. 💼 Top Matching Jobs")

        if jobs.empty or description_col is None:
            st.error(
                "Jobs could not be loaded. Check the job CSV file "
                "and its job description column."
            )

        else:
            try:
                vectorizer = TfidfVectorizer(
                    stop_words="english",
                    max_features=20000
                )

                job_matrix = vectorizer.fit_transform(
                    jobs[description_col].fillna("").astype(str)
                )
                resume_vector = vectorizer.transform([resume_text])

                scores = cosine_similarity(
                    resume_vector, job_matrix
                ).flatten()

                results = jobs.copy()
                results["Match Score (%)"] = (scores * 100).round(2)

                results = results.sort_values(
                    "Match Score (%)", ascending=False
                ).head(5).copy()

                display_columns = [
                    title_col,
                    "display_company",
                    category_col,
                    "Match Score (%)"
                ]

                display_results = results[display_columns].rename(
                    columns={
                        title_col: "Job Title",
                        "display_company": "Company",
                        category_col: "Job Category"
                    }
                )

                st.dataframe(
                    display_results.reset_index(drop=True),
                    use_container_width=True
                )

                best_job = results.iloc[0]

                st.subheader("🏆 Best Matching Job")
                st.write(f"**Job Title:** {best_job[title_col]}")
                st.write(
                    f"**Company:** {best_job['display_company']}"
                )
                st.write(
                    f"**Job Category:** {best_job[category_col]}"
                )
                st.write(
                    f"**Text Similarity:** "
                    f"{best_job['Match Score (%)']}%"
                )

                st.caption(
                    "Match score measures text similarity; it is not "
                    "the probability of getting hired."
                )

                # -------- 3. MISSING SKILLS --------
                st.header("3. 📚 Resume Skills & Missing Skills")

                resume_skills = extract_skills(resume_text)
                job_description = str(best_job[description_col])
                required_skills = extract_skills(job_description)

                missing_skills = [
                    skill for skill in required_skills
                    if skill not in resume_skills
                ]

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("✅ Skills Found in Resume")
                    if resume_skills:
                        st.write(", ".join(resume_skills))
                    else:
                        st.info(
                            "No skills from the configured technical "
                            "skills list were detected."
                        )

                with col2:
                    st.subheader("📌 Skills Required by Best Job")
                    if required_skills:
                        st.write(", ".join(required_skills))
                    else:
                        st.info(
                            "No configured technical skills were "
                            "detected in this job description."
                        )

                st.subheader("🎯 Missing Skills to Learn")
                if missing_skills:
                    st.write(", ".join(missing_skills))
                    st.info(
                        "Consider learning these skills for the "
                        "best-matching job."
                    )
                else:
                    st.success(
                        "No missing skills from the configured list "
                        "were detected for this job."
                    )

                st.caption(
                    "Skill analysis checks a predefined list of "
                    "technical skills and may miss synonyms or skills "
                    "not included in that list."
                )

            except Exception as error:
                st.error(f"Job matching failed: {error}")

st.divider()
st.caption(
    "SmartHire | Resume Classification, Job Matching "
    "& Skill-Gap Analysis Prototype"
)
