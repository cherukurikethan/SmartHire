
"""Train and save a resume category classifier."""

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

from src import config


def train_classifier():
    # Load cleaned resume data
    df = pd.read_csv(config.RESUMES_CLEAN_CSV)
    df = df.dropna(subset=["Category", "text"])

    X = df["text"].astype(str)
    y = df["Category"].astype(str)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # TF-IDF + Linear SVM classifier
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            stop_words="english"
        )),
        ("classifier", LinearSVC())
    ])

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Save trained model
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODEL_DIR / "resume_classifier.joblib"
    joblib.dump(model, model_path)

    print("Resume classifier trained successfully!")
    print(f"Training resumes: {len(X_train)}")
    print(f"Testing resumes: {len(X_test)}")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Model saved to: {model_path}")

    return model


if __name__ == "__main__":
    train_classifier()
