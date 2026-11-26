
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    brier_score_loss,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

from sklearn.calibration import calibration_curve


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR / "artifacts" / "preprocessing"
    / "kidney_processed_data.npz"
)

MODEL_PATH = (
    BASE_DIR / "artifacts" / "models"
    / "kidney_best_model.joblib"
)

VALIDATION_DIR = (
    BASE_DIR / "artifacts" / "validation"
)

VALIDATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# 2. LOAD MODEL AND TEST DATA
# ==================================================

def load_validation_data():

    print("\nLoading trained model and test data...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    with np.load(DATA_PATH) as data:

        X_test = data["X_test"]
        y_test = data["y_test"].astype(int)

    if not np.isfinite(X_test).all():
        raise ValueError(
            "Test data contains invalid values."
        )

    if set(np.unique(y_test)) != {0, 1}:
        raise ValueError(
            "Test data must contain both classes."
        )

    if hasattr(model, "n_features_in_"):
        if X_test.shape[1] != model.n_features_in_:
            raise ValueError(
                "Model and test feature dimensions do not match."
            )

    print("Model loaded successfully.")
    print("Test samples:", len(X_test))
    print("Test features:", X_test.shape[1])

    return model, X_test, y_test


# ==================================================
# 3. CALCULATE VALIDATION METRICS
# ==================================================

def calculate_metrics(model, X_test, y_test):

    print("\nGenerating predictions...")

    y_pred = model.predict(X_test)

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "Model does not support probability predictions."
        )

    probabilities = model.predict_proba(X_test)

    positive_index = list(model.classes_).index(1)

    y_probability = probabilities[:, positive_index]

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    ).ravel()

    # Sensitivity
    sensitivity = tp / (tp + fn) if (tp + fn) else 0

    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) else 0

    # Evaluation metrics
    metrics = {

        "Accuracy": float(
            accuracy_score(y_test, y_pred)
        ),

        "Balanced_Accuracy": float(
            balanced_accuracy_score(y_test, y_pred)
        ),

        "Precision": float(
            precision_score(
                y_test, y_pred, zero_division=0
            )
        ),

        "Sensitivity": float(sensitivity),

        "Specificity": float(specificity),

        "F1_Score": float(
            f1_score(
                y_test, y_pred, zero_division=0
            )
        ),

        "ROC_AUC": float(
            roc_auc_score(y_test, y_probability)
        ),

        "PR_AUC": float(
            average_precision_score(
                y_test, y_probability
            )
        ),

        "MCC": float(
            matthews_corrcoef(y_test, y_pred)
        ),

        "Brier_Score": float(
            brier_score_loss(y_test, y_probability)
        )
    }

    print("\n" + "=" * 60)
    print("EARLYGUARD - MODEL VALIDATION RESULTS")
    print("=" * 60)

    for name, value in metrics.items():

        print(f"{name}: {value:.4f}")

    print("\nConfusion Matrix")

    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )
    )

    print("\nClassification Report")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Not CKD", "CKD"],
            zero_division=0
        )
    )

    return metrics, y_pred, y_probability


# ==================================================
# 4. BOOTSTRAP CONFIDENCE INTERVALS
# ==================================================

def calculate_confidence_intervals(
    y_test,
    y_pred,
    y_probability,
    iterations=1000
):

    print("\nCalculating bootstrap confidence intervals...")

    rng = np.random.default_rng(42)

    negative_indices = np.where(y_test == 0)[0]
    positive_indices = np.where(y_test == 1)[0]

    results = {
        "ROC_AUC": [],
        "Sensitivity": [],
        "Specificity": []
    }

    for _ in range(iterations):

        sampled_negative = rng.choice(
            negative_indices,
            size=len(negative_indices),
            replace=True
        )

        sampled_positive = rng.choice(
            positive_indices,
            size=len(positive_indices),
            replace=True
        )

        indices = np.concatenate([
            sampled_negative,
            sampled_positive
        ])

        actual = y_test[indices]
        predicted = y_pred[indices]
        probability = y_probability[indices]

        tn, fp, fn, tp = confusion_matrix(
            actual,
            predicted,
            labels=[0, 1]
        ).ravel()

        results["ROC_AUC"].append(
            roc_auc_score(actual, probability)
        )

        results["Sensitivity"].append(
            tp / (tp + fn) if tp + fn else 0
        )

        results["Specificity"].append(
            tn / (tn + fp) if tn + fp else 0
        )

    intervals = {}

    for name, values in results.items():

        lower, upper = np.percentile(
            values,
            [2.5, 97.5]
        )

        intervals[name] = {
            "Lower_95_CI": float(lower),
            "Upper_95_CI": float(upper)
        }

    with open(
        VALIDATION_DIR / "confidence_intervals.json",
        "w"
    ) as file:

        json.dump(
            intervals,
            file,
            indent=4
        )

    print("\n95% Confidence Intervals")

    for name, interval in intervals.items():

        print(
            f"{name}: "
            f"{interval['Lower_95_CI']:.4f} - "
            f"{interval['Upper_95_CI']:.4f}"
        )

    return intervals


# ==================================================
# 5. GENERATE VALIDATION GRAPHS
# ==================================================

def generate_validation_graphs(
    y_test,
    y_pred,
    y_probability
):

    print("\nGenerating validation graphs...")

    # ----------------------------------------------
    # A. CONFUSION MATRIX
    # ----------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not CKD", "CKD"]
    ).plot(
        ax=ax,
        cmap="Blues",
        values_format="d"
    )

    ax.set_title("EarlyGuard - Validation Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        VALIDATION_DIR / "validation_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # ----------------------------------------------
    # B. ROC CURVE
    # ----------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 6))

    RocCurveDisplay.from_predictions(
        y_test,
        y_probability,
        name="EarlyGuard",
        ax=ax
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax.set_title("EarlyGuard - Validation ROC Curve")

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        VALIDATION_DIR / "validation_roc_curve.png",
        dpi=300
    )

    plt.close()

    # ----------------------------------------------
    # C. CALIBRATION CURVE
    # ----------------------------------------------

    fraction_positive, mean_probability = (
        calibration_curve(
            y_test,
            y_probability,
            n_bins=5,
            strategy="quantile"
        )
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(
        mean_probability,
        fraction_positive,
        marker="o",
        label="EarlyGuard"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect Calibration"
    )

    ax.set_xlabel("Mean Predicted Probability")

    ax.set_ylabel("Fraction of Positive Cases")

    ax.set_title("EarlyGuard - Calibration Curve")

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        VALIDATION_DIR / "calibration_curve.png",
        dpi=300
    )

    plt.close()

    print("Validation graphs saved successfully.")


# ==================================================
# 6. SAVE VALIDATION REPORT
# ==================================================

def save_validation_report(
    metrics,
    intervals,
    y_test,
    y_pred,
    y_probability
):

    report = {
        "project": "EarlyGuard",
        "task": "Chronic Kidney Disease Prediction",
        "test_samples": int(len(y_test)),
        "metrics": metrics,
        "confidence_intervals": intervals,
        "note": (
            "Evaluation on the existing held-out test set. "
            "Not an independent external clinical validation."
        )
    }

    with open(
        VALIDATION_DIR / "validation_report.json",
        "w"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    predictions = pd.DataFrame({

        "Actual": y_test,
        "Predicted": y_pred,
        "CKD_Probability": y_probability

    })

    predictions.to_csv(
        VALIDATION_DIR / "validation_predictions.csv",
        index=False
    )

    print("\nValidation report saved.")


# ==================================================
# 7. MAIN EXECUTION
# ==================================================

def main():

    print("\n")
    print("=" * 65)
    print("EARLYGUARD - MODEL VALIDATION")
    print("=" * 65)

    model, X_test, y_test = load_validation_data()

    metrics, y_pred, y_probability = calculate_metrics(
        model,
        X_test,
        y_test
    )

    intervals = calculate_confidence_intervals(
        y_test,
        y_pred,
        y_probability
    )

    generate_validation_graphs(
        y_test,
        y_pred,
        y_probability
    )

    save_validation_report(
        metrics,
        intervals,
        y_test,
        y_pred,
        y_probability
    )

    print("\n" + "=" * 65)
    print("EARLYGUARD VALIDATION COMPLETED!")
    print("=" * 65)

    print("\nValidation files saved in:")
    print(VALIDATION_DIR)


if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\nVALIDATION FAILED!")
        print("Error:", error)

        raise
