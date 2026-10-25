
import streamlit as st
import pandas as pd
import joblib
import fitz
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="SmartHire",
    page_icon="💼",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "resume_classifier.joblib"
JOB_PATH = ROOT / "data" / "raw" / "naukri_com-job_sample.csv"

st.title("💼 SmartHire")
st.subheader("Resume-to-Job Matching & Career Guidance Engine")
st.write("Upload your resume or paste your resume text to explore matching jobs.")

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data
def load_jobs():
    if not JOB_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(JOB_PATH)

    possible_columns = [
        "jobdescription",
        "job_description",
        "Job Description",
        "description"
    ]

    description_col = next(
        (col for col in possible_columns if col in df.columns),
        None
    )

    title_col = next(
        (col for col in ["jobtitle", "Job Title", "title", "Title"]
         if col in df.columns),
        None
    )

    if description_col is None:
        return pd.DataFrame()

    df = df.dropna(subset=[description_col]).copy()
    df["description_text"] = df[description_col].astype(str)

    if title_col:
        df["display_title"] = df[title_col].fillna("Job").astype(str)
    else:
        df["display_title"] = "Job"

    return df.reset_index(drop=True)

model = load_model()
jobs = load_jobs()

uploaded_file = st.file_uploader(
    "Upload resume PDF",
    type=["pdf"]
)

resume_text = ""

if uploaded_file:
    try:
        with fitz.open(
            stream=uploaded_file.getvalue(),
            filetype="pdf"
        ) as pdf:
            resume_text = "\n".join(
                page.get_text() for page in pdf
            )

        if resume_text.strip():
            st.success("Resume text extracted successfully!")
            with st.expander("View extracted text"):
                st.text(resume_text)
        else:
            st.warning(
                "No selectable text found in this PDF. "
                "Try a text-based PDF or paste your resume."
            )
    except Exception as error:
        st.error(f"Could not read PDF: {error}")

manual_text = st.text_area(
    "Or paste your resume text",
    height=180,
    placeholder="Python, SQL, Machine Learning, Data Analysis..."
)

if manual_text.strip():
    resume_text = manual_text

if st.button("Analyze Resume", type="primary"):
    if not resume_text.strip():
        st.warning("Please upload a PDF or paste resume text.")
    else:
        st.header("1. Resume Category")

        if model is not None:
            try:
                category = model.predict([resume_text])[0]
                st.success(f"Predicted Category: {category}")
            except Exception as error:
                st.error(f"Classification failed: {error}")
        else:
            st.warning(
                "Trained model not found. "
                "Check models/resume_classifier.joblib."
            )

        st.header("2. Top Matching Jobs")

        if jobs.empty:
            st.warning(
                "Jobs could not be loaded. "
                "Check the CSV file and its description column."
            )
        else:
            try:
                vectorizer = TfidfVectorizer(
                    stop_words="english",
                    max_features=20000
                )

                job_matrix = vectorizer.fit_transform(
                    jobs["description_text"]
                )

                resume_vector = vectorizer.transform([resume_text])

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

                st.dataframe(
                    results[
                        ["display_title", "Match Score (%)"]
                    ].rename(columns={
                        "display_title": "Job Title"
                    }),
                    use_container_width=True
                )

                best_job = results.iloc[0]
                st.subheader("🎯 Best Matching Job")
                st.write(best_job["display_title"])
                st.write(
                    f"Text similarity: "
                    f"{best_job['Match Score (%)']}%"
                )

                st.caption(
                    "Match score is text similarity, not a "
                    "probability of getting hired."
                )

            except Exception as error:
                st.error(f"Job matching failed: {error}")

        st.header("3. Resume Skills")
        st.write(
            "Review your resume against the skills required "
            "for your target job. The current matching score "
            "compares text similarity; it does not verify "
            "all required skills."
        )

st.divider()
st.caption("SmartHire — Resume analysis and job matching prototype")
