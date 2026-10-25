
from pathlib import Path
import joblib

# Find the project folder
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "resume_classifier.joblib"

# Load the trained model
model = joblib.load(MODEL_PATH)

# Enter resume text
resume_text = input("Paste your resume text here: ")

# Predict the resume category
predicted_category = model.predict([resume_text])[0]

print("\nPredicted Resume Category:", predicted_category)
