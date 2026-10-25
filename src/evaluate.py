
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def evaluate_classifier(y_true, y_pred):
    """Calculate and return common classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": classification_report(
            y_true, y_pred, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
