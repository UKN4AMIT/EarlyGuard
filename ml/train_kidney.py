
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

warnings.filterwarnings("ignore")


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR / "artifacts" / "preprocessing"
    / "kidney_processed_data.npz"
)

MODEL_DIR = BASE_DIR / "artifacts" / "models"
EVALUATION_DIR = BASE_DIR / "artifacts" / "evaluation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD PROCESSED DATA
# ==================================================

def load_processed_data():

    print("\nLoading processed CKD dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}"
        )

    data = np.load(DATA_PATH, allow_pickle=False)

    X_train = data["X_train"]
    X_test = data["X_test"]

    y_train = data["y_train"].astype(int)
    y_test = data["y_test"].astype(int)

    if not np.isfinite(X_train).all():
        raise ValueError("Invalid values in training dataset.")

    if not np.isfinite(X_test).all():
        raise ValueError("Invalid values in testing dataset.")

    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError("Training data must contain both classes.")

    print("\nDataset loaded successfully.")

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    return X_train, X_test, y_train, y_test


# ==================================================
# 3. DEFINE MACHINE LEARNING MODELS
# ==================================================

def get_models():

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=4,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42
        )
    }

    return models


# ==================================================
# 4. CROSS VALIDATION AND MODEL COMPARISON
# ==================================================

def compare_models(X_train, y_train):

    print("\n" + "=" * 65)
    print("EARLYGUARD - MODEL COMPARISON")
    print("=" * 65)

    models = get_models()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = []

    for name, model in models.items():

        print(f"\nEvaluating: {name}")

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        )

        mean_auc = float(scores.mean())
        std_auc = float(scores.std())

        results.append({
            "Model": name,
            "Mean CV ROC-AUC": mean_auc,
            "Std CV ROC-AUC": std_auc
        })

        print("Fold ROC-AUC:", np.round(scores, 4))
        print(f"Mean ROC-AUC: {mean_auc:.4f}")
        print(f"Std ROC-AUC: {std_auc:.4f}")

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Mean CV ROC-AUC",
        ascending=False
    ).reset_index(drop=True)

    results_df.to_csv(
        EVALUATION_DIR / "model_comparison.csv",
        index=False
    )

    print("\nMODEL COMPARISON RESULTS")
    print(results_df.to_string(index=False))

    return models, results_df


# ==================================================
# 5. TRAIN BEST MODEL
# ==================================================

def train_best_model(
    models,
    results_df,
    X_train,
    y_train
):

    best_model_name = results_df.iloc[0]["Model"]

    print("\n" + "=" * 65)
    print("BEST MODEL SELECTION")
    print("=" * 65)

    print("Selected model:", best_model_name)

    best_model = models[best_model_name]

    print("\nTraining best model on complete training data...")

    best_model.fit(
        X_train,
        y_train
    )

    print("Model training completed.")

    return best_model, best_model_name


# ==================================================
# 6. MODEL EVALUATION
# ==================================================

def evaluate_model(
    model,
    model_name,
    X_test,
    y_test
):

    print("\n" + "=" * 65)
    print("EARLYGUARD - MODEL EVALUATION")
    print("=" * 65)

    # Predictions
    y_pred = model.predict(X_test)

    # Prediction probabilities for positive class (CKD)
    y_probability = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    metrics = {

        "Model": model_name,

        "Accuracy": float(
            accuracy_score(y_test, y_pred)
        ),

        "Precision": float(
            precision_score(
                y_test,
                y_pred,
                zero_division=0
            )
        ),

        "Recall_Sensitivity": float(
            recall_score(
                y_test,
                y_pred,
                zero_division=0
            )
        ),

        "F1_Score": float(
            f1_score(
                y_test,
                y_pred,
                zero_division=0
            )
        ),

        "ROC_AUC": float(
            roc_auc_score(
                y_test,
                y_probability
            )
        )
    }

    print("\nEvaluation Metrics")

    for metric, value in metrics.items():

        if metric != "Model":
            print(f"{metric}: {value:.4f}")

    # Classification report
    print("\nClassification Report")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Not CKD", "CKD"],
            zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    print("\nConfusion Matrix")
    print(cm)

    # Save metrics
    with open(
        EVALUATION_DIR / "best_model_metrics.json",
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # Save predictions
    prediction_df = pd.DataFrame({

        "Actual": y_test,
        "Predicted": y_pred,
        "CKD_Probability": y_probability

    })

    prediction_df.to_csv(
        EVALUATION_DIR / "test_predictions.csv",
        index=False
    )

    return metrics, y_pred, y_probability, cm


# ==================================================
# 7. GENERATE EVALUATION GRAPHS
# ==================================================

def generate_evaluation_graphs(
    model,
    results_df,
    y_test,
    y_pred,
    y_probability,
    cm
):

    print("\nGenerating evaluation graphs...")

    # ----------------------------------------------
    # A. MODEL COMPARISON GRAPH
    # ----------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.bar(
        results_df["Model"],
        results_df["Mean CV ROC-AUC"]
    )

    plt.title("EarlyGuard - Model Comparison")

    plt.xlabel("Machine Learning Models")
    plt.ylabel("Mean Cross-Validation ROC-AUC")

    plt.ylim(0, 1)

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        EVALUATION_DIR / "model_comparison.png",
        dpi=300
    )

    plt.close()

    # ----------------------------------------------
    # B. CONFUSION MATRIX
    # ----------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not CKD", "CKD"]
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d"
    )

    plt.title("EarlyGuard - Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        EVALUATION_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # ----------------------------------------------
    # C. ROC CURVE
    # ----------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 6))

    RocCurveDisplay.from_predictions(
        y_test,
        y_probability,
        name="Best Model",
        ax=ax
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax.set_title("EarlyGuard - ROC Curve")

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        EVALUATION_DIR / "roc_curve.png",
        dpi=300
    )

    plt.close()

    print("All evaluation graphs saved.")


# ==================================================
# 8. SAVE TRAINED MODEL
# ==================================================

def save_model(model, model_name):

    model_path = MODEL_DIR / "kidney_best_model.joblib"

    joblib.dump(
        model,
        model_path
    )

    with open(
        MODEL_DIR / "model_info.json",
        "w"
    ) as file:

        json.dump(
            {
                "model_name": model_name,
                "target": "Chronic Kidney Disease",
                "positive_class": 1,
                "negative_class": 0,
                "selection_metric": "Mean 5-fold CV ROC-AUC"
            },
            file,
            indent=4
        )

    print("\nModel saved successfully:")
    print(model_path)


# ==================================================
# 9. MAIN EXECUTION
# ==================================================

def main():

    print("\n")
    print("=" * 65)
    print("WELCOME TO EARLYGUARD")
    print("CHRONIC KIDNEY DISEASE PREDICTION SYSTEM")
    print("=" * 65)

    # Step 1: Load data
    X_train, X_test, y_train, y_test = (
        load_processed_data()
    )

    # Step 2: Compare models
    models, results_df = compare_models(
        X_train,
        y_train
    )

    # Step 3: Train best model
    best_model, best_model_name = train_best_model(
        models,
        results_df,
        X_train,
        y_train
    )

    # Step 4: Evaluate model
    metrics, y_pred, y_probability, cm = evaluate_model(
        best_model,
        best_model_name,
        X_test,
        y_test
    )

    # Step 5: Generate graphs
    generate_evaluation_graphs(
        best_model,
        results_df,
        y_test,
        y_pred,
        y_probability,
        cm
    )

    # Step 6: Save model
    save_model(
        best_model,
        best_model_name
    )

    # Final summary
    print("\n")
    print("=" * 65)
    print("EARLYGUARD MODEL TRAINING COMPLETED!")
    print("=" * 65)

    print("\nBest Model:", best_model_name)

    print(
        "Test Accuracy:",
        round(metrics["Accuracy"] * 100, 2),
        "%"
    )

    print(
        "Test Sensitivity:",
        round(metrics["Recall_Sensitivity"] * 100, 2),
        "%"
    )

    print(
        "Test ROC-AUC:",
        round(metrics["ROC_AUC"], 4)
    )

    print("\nModels saved in:", MODEL_DIR)
    print("Graphs saved in:", EVALUATION_DIR)

    print("\nNEXT PHASE: MODEL EXPLAINABILITY (SHAP)")


# ==================================================
# 10. EXECUTION
# ==================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print("\nMODEL TRAINING FAILED!")
        print("Error:", error)

        raise
