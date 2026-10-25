
import streamlit as st
import pandas as pd
import fitz
import joblib

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
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "resume_classifier.joblib"
DATA_PATH = ROOT / "data" / "raw" / "naukri_com-job_sample.csv"

# ---------------- SKILLS LIST ----------------
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

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


# ---------------- LOAD JOB DATA ----------------
@st.cache_data
def load_jobs():
    if not DATA_PATH.exists():
        return pd.DataFrame(), None, None

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()

    description_candidates = [
        "jobdescription",
        "job_description",
        "Job Description",
        "description"
    ]

    title_candidates = [
        "Job Title",
        "jobtitle",
        "Title",
        "title"
    ]

    company_candidates = [
        "Company",
        "company",
        "Company Name",
        "company_name"
    ]

    description_col = next(
        (col for col in description_candidates if col in df.columns),
        None
    )

    title_col = next(
        (col for col in title_candidates if col in df.columns),
        None
    )

    company_col = next(
        (col for col in company_candidates if col in df.columns),
        None
    )

    if description_col is None:
        return pd.DataFrame(), None, None

    df = df.dropna(subset=[description_col]).copy()
    df[description_col] = df[description_col].astype(str)

    if title_col is None:
        df["Display Job Title"] = "Job"
        title_col = "Display Job Title"

    if company_col is None:
        df["Display Company"] = "Not specified"
        company_col = "Display Company"

    df = df.reset_index(drop=True)

    return df, description_col, (title_col, company_col)


# ---------------- APP HEADER ----------------
st.title("💼 SmartHire")
st.subheader("Resume-to-Job Matching & Career Guidance Engine")

st.write(
    "Upload your resume or paste your resume text to predict "
    "its category, discover matching jobs, and explore technical skills."
)

st.divider()

# ---------------- RESUME INPUT ----------------
uploaded_file = st.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"]
)

manual_text = st.text_area(
    "Or paste your resume text here",
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
            pdf_text = "\n".join(
                page.get_text() for page in pdf
            )

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
        st.error(f"Could not read PDF: {error}")

# ---------------- ANALYSIS ----------------
if st.button("Analyze Resume", type="primary"):

    if not resume_text.strip():
        st.warning("Please upload a PDF or paste resume text.")

    else:
        # -------- CATEGORY PREDICTION --------
        st.header("1. 🎯 Resume Category")

        model = load_model()

        if model is not None:
            try:
                category = model.predict([resume_text])[0]
                st.success(f"Predicted Category: {category}")

            except Exception as error:
                st.error(f"Category prediction failed: {error}")

        else:
            st.warning(
                "Trained model not found. "
                "Check models/resume_classifier.joblib."
            )

        # -------- JOB RECOMMENDATIONS --------
        st.header("2. 💼 Top Matching Jobs")

        jobs, description_col, display_columns = load_jobs()

        if jobs.empty or description_col is None:
            st.error(
                "Could not load jobs. Check the CSV file and "
                "its job description column."
            )

        else:
            try:
                title_col, company_col = display_columns

                vectorizer = TfidfVectorizer(
                    stop_words="english",
                    max_features=20000
                )

                job_matrix = vectorizer.fit_transform(
                    jobs[description_col]
                )

                resume_vector = vectorizer.transform(
                    [resume_text]
                )

                scores = cosine_similarity(
                    resume_vector,
                    job_matrix
                ).flatten()

                results = jobs.copy()
                results["Match Score (%)"] = (
                    scores * 100
                ).round(2)

                results = results.sort_values(
                    "Match Score (%)",
                    ascending=False
                ).head(5)

                display_results = results[
                    [
                        title_col,
                        company_col,
                        "Match Score (%)"
                    ]
                ].rename(
                    columns={
                        title_col: "Job Title",
                        company_col: "Company"
                    }
                )

                st.dataframe(
                    display_results.reset_index(drop=True),
                    use_container_width=True
                )

                best_job = results.iloc[0]

                st.subheader("🏆 Best Matching Job")
                st.write(f"**{best_job[title_col]}**")
                st.write(f"Company: {best_job[company_col]}")
                st.write(
                    f"Text similarity: "
                    f"{best_job['Match Score (%)']}%"
                )

                st.caption(
                    "Match score measures text similarity. "
                    "It is not the probability of getting hired."
                )

                # -------- SKILL ANALYSIS --------
                st.header("3. 📚 Resume Skills")

                resume_lower = resume_text.lower()
                job_lower = str(
                    best_job[description_col]
                ).lower()

                resume_skills = [
                    skill for skill in TECHNICAL_SKILLS
                    if skill.lower() in resume_lower
                ]

                required_skills = [
                    skill for skill in TECHNICAL_SKILLS
                    if skill.lower() in job_lower
                ]

                missing_skills = [
                    skill for skill in required_skills
                    if skill not in resume_skills
                ]

                st.write("**Skills Found in Resume:**")

                if resume_skills:
                    st.write(", ".join(resume_skills))
                else:
                    st.write("No listed technical skills detected.")

                st.write("**Skills to Explore:**")

                if missing_skills:
                    st.write(", ".join(missing_skills))
                else:
                    st.write(
                        "No missing skills from the configured "
                        "technical skills list were detected."
                    )

                st.caption(
                    "Skill detection checks a predefined list. "
                    "It may miss skills expressed using other names."
                )

            except Exception as error:
                st.error(f"Job matching failed: {error}")

st.divider()
st.caption(
    "SmartHire | Resume Classification, Job Matching "
    "& Career Guidance Prototype"
)
