
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "artifacts" / "models"

PREPROCESSING_DIR = (
    BASE_DIR / "artifacts" / "preprocessing"
)

EXPLAINABILITY_DIR = (
    BASE_DIR / "artifacts" / "explainability"
)

MODEL_PATH = MODEL_DIR / "kidney_best_model.joblib"

PREPROCESSOR_PATH = (
    PREPROCESSING_DIR / "kidney_preprocessor.joblib"
)

DATA_PATH = (
    PREPROCESSING_DIR / "kidney_processed_data.npz"
)

EXPLAINABILITY_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MAX_EXPLAIN_SAMPLES = 80
BACKGROUND_SAMPLES = 100


# ==================================================
# 2. LOAD SAVED FILES
# ==================================================

def load_saved_files():

    print("\n" + "=" * 60)
    print("EARLYGUARD - SHAP EXPLAINABILITY")
    print("=" * 60)

    required_files = [
        MODEL_PATH,
        PREPROCESSOR_PATH,
        DATA_PATH
    ]

    for path in required_files:

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    print("\nLoading trained model...")

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully.")

    print("\nLoading preprocessor...")

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    print("Preprocessor loaded successfully.")

    print("\nLoading processed dataset...")

    data = np.load(
        DATA_PATH,
        allow_pickle=False
    )

    X_train = np.asarray(
        data["X_train"],
        dtype=float
    )

    X_test = np.asarray(
        data["X_test"],
        dtype=float
    )

    y_test = np.asarray(
        data["y_test"],
        dtype=int
    )

    print("Dataset loaded successfully.")

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    return model, preprocessor, X_train, X_test, y_test


# ==================================================
# 3. EXTRACT FEATURE NAMES
# ==================================================

def get_feature_names(preprocessor, feature_count):

    print("\nExtracting feature names...")

    try:

        feature_names = list(
            preprocessor.get_feature_names_out()
        )

    except Exception as error:

        raise RuntimeError(
            "Could not extract feature names "
            "from the saved preprocessor."
        ) from error

    if len(feature_names) != feature_count:

        raise ValueError(
            f"Feature mismatch: Model expects "
            f"{feature_count} features, but "
            f"preprocessor returned "
            f"{len(feature_names)} names."
        )

    print("Total features:", len(feature_names))

    return feature_names


# ==================================================
# 4. CREATE SHAP EXPLAINER
# ==================================================

def create_shap_explainer(model, X_train):

    print("\nCreating SHAP explainer...")

    rng = np.random.default_rng(42)

    background_size = min(
        BACKGROUND_SAMPLES,
        len(X_train)
    )

    background_indices = rng.choice(
        len(X_train),
        size=background_size,
        replace=False
    )

    background = X_train[background_indices]

    masker = shap.maskers.Independent(
        background
    )

    explainer = shap.Explainer(
        model.predict_proba,
        masker,
        algorithm="permutation"
    )

    print("SHAP explainer created successfully.")

    return explainer


# ==================================================
# 5. CALCULATE SHAP VALUES
# ==================================================

def calculate_shap_values(
    explainer,
    model,
    X_test,
    feature_names
):

    print("\nCalculating SHAP values...")

    sample_count = min(
        MAX_EXPLAIN_SAMPLES,
        len(X_test)
    )

    X_explain = X_test[:sample_count]

    feature_count = X_explain.shape[1]

    shap_explanation = explainer(
        X_explain,
        max_evals=(2 * feature_count + 1)
    )

    all_values = np.asarray(
        shap_explanation.values
    )

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            "CKD positive class (1) not found."
        )

    positive_index = classes.index(1)

    if all_values.ndim == 3:

        positive_values = all_values[
            :, :, positive_index
        ]

    elif all_values.ndim == 2:

        positive_values = all_values

    else:

        raise ValueError(
            f"Unexpected SHAP output shape: "
            f"{all_values.shape}"
        )

    base_values = np.asarray(
        shap_explanation.base_values
    )

    if base_values.ndim == 2:

        positive_base_values = base_values[
            :, positive_index
        ]

    elif base_values.ndim == 1:

        positive_base_values = base_values

    elif base_values.ndim == 0:

        positive_base_values = np.full(
            sample_count,
            float(base_values)
        )

    else:

        raise ValueError(
            "Unexpected SHAP base value shape."
        )

    if positive_values.shape[1] != len(feature_names):

        raise ValueError(
            "SHAP feature count does not match "
            "the preprocessed dataset."
        )

    print("SHAP calculation completed.")

    print("Explained samples:", sample_count)

    print("SHAP values shape:", positive_values.shape)

    return (
        X_explain,
        positive_values,
        positive_base_values
    )


# ==================================================
# 6. GLOBAL FEATURE IMPORTANCE
# ==================================================

def generate_feature_importance(
    shap_values,
    feature_names
):

    print("\nGenerating global feature importance...")

    importance = np.mean(
        np.abs(shap_values),
        axis=0
    )

    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Mean_Absolute_SHAP": importance

    })

    importance_df = importance_df.sort_values(
        by="Mean_Absolute_SHAP",
        ascending=False
    )

    importance_df.to_csv(
        EXPLAINABILITY_DIR / "feature_importance.csv",
        index=False
    )

    print("\nTOP 10 IMPORTANT FEATURES")

    print(
        importance_df.head(10).to_string(
            index=False
        )
    )

    return importance_df


# ==================================================
# 7. GENERATE SHAP SUMMARY PLOT
# ==================================================

def generate_summary_plot(
    shap_values,
    X_explain,
    feature_names
):

    print("\nGenerating SHAP summary plot...")

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        show=False,
        max_display=20
    )

    plt.title(
        "EarlyGuard - SHAP Summary Plot"
    )

    plt.tight_layout()

    plt.savefig(
        EXPLAINABILITY_DIR / "shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Summary plot saved.")


# ==================================================
# 8. GENERATE SHAP BAR PLOT
# ==================================================

def generate_bar_plot(
    shap_values,
    X_explain,
    feature_names
):

    print("\nGenerating SHAP bar plot...")

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=20
    )

    plt.title(
        "EarlyGuard - Global Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        EXPLAINABILITY_DIR / "shap_feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Feature importance plot saved.")


# ==================================================
# 9. INDIVIDUAL PATIENT EXPLANATION
# ==================================================

def generate_patient_explanation(
    model,
    X_explain,
    shap_values,
    base_values,
    feature_names
):

    print("\nGenerating individual patient explanation...")

    patient_index = 0

    patient_data = X_explain[patient_index]

    patient_shap = shap_values[patient_index]

    prediction = model.predict(
        patient_data.reshape(1, -1)
    )[0]

    probability = model.predict_proba(
        patient_data.reshape(1, -1)
    )[0]

    classes = list(model.classes_)

    positive_index = classes.index(1)

    ckd_probability = probability[positive_index]

    print("\nPATIENT PREDICTION")
    print("-" * 40)

    print(
        "Prediction:",
        "CKD" if prediction == 1 else "Not CKD"
    )

    print(
        f"CKD Probability: {ckd_probability * 100:.2f}%"
    )

    patient_df = pd.DataFrame({

        "Feature": feature_names,

        "Value": patient_data,

        "SHAP_Contribution": patient_shap

    })

    patient_df = patient_df.sort_values(
        by="SHAP_Contribution",
        key=abs,
        ascending=False
    )

    patient_df.to_csv(
        EXPLAINABILITY_DIR / "patient_explanation.csv",
        index=False
    )

    print("\nTOP PATIENT CONTRIBUTIONS")

    print(
        patient_df.head(10).to_string(
            index=False
        )
    )

    # Waterfall explanation

    waterfall_explanation = shap.Explanation(

        values=patient_shap,

        base_values=base_values[patient_index],

        data=patient_data,

        feature_names=feature_names

    )

    plt.figure(figsize=(12, 8))

    shap.plots.waterfall(
        waterfall_explanation,
        max_display=15,
        show=False
    )

    plt.title(
        "EarlyGuard - Individual Patient Explanation"
    )

    plt.tight_layout()

    plt.savefig(
        EXPLAINABILITY_DIR / "patient_waterfall.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nPatient waterfall plot saved.")

    return prediction, ckd_probability


# ==================================================
# 10. SAVE EXPLAINABILITY SUMMARY
# ==================================================

def save_summary(
    importance_df,
    X_explain,
    shap_values
):

    summary = {

        "project": "EarlyGuard",

        "target": "Chronic Kidney Disease",

        "explained_samples": int(len(X_explain)),

        "total_features": int(X_explain.shape[1]),

        "top_features": (
            importance_df.head(10)["Feature"].tolist()
        ),

        "mean_absolute_shap": (
            importance_df.head(10)[
                "Mean_Absolute_SHAP"
            ].tolist()
        )

    }

    with open(
        EXPLAINABILITY_DIR / "shap_summary.json",
        "w"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print("\nSHAP summary saved.")


# ==================================================
# 11. MAIN EXECUTION
# ==================================================

def main():

    print("\n")

    print("=" * 60)

    print("EARLYGUARD - EXPLAINABLE AI MODULE")

    print("=" * 60)

    # Step 1: Load saved files

    (
        model,
        preprocessor,
        X_train,
        X_test,
        y_test
    ) = load_saved_files()

    # Step 2: Validate model dimensions

    if X_train.shape[1] != X_test.shape[1]:

        raise ValueError(
            "Training and testing feature dimensions differ."
        )

    # Step 3: Extract feature names

    feature_names = get_feature_names(
        preprocessor,
        X_train.shape[1]
    )

    # Step 4: Create explainer

    explainer = create_shap_explainer(
        model,
        X_train
    )

    # Step 5: Calculate SHAP values

    (
        X_explain,
        shap_values,
        base_values
    ) = calculate_shap_values(
        explainer,
        model,
        X_test,
        feature_names
    )

    # Step 6: Global feature importance

    importance_df = generate_feature_importance(
        shap_values,
        feature_names
    )

    # Step 7: Summary plot

    generate_summary_plot(
        shap_values,
        X_explain,
        feature_names
    )

    # Step 8: Bar plot

    generate_bar_plot(
        shap_values,
        X_explain,
        feature_names
    )

    # Step 9: Individual patient explanation

    prediction, probability = generate_patient_explanation(
        model,
        X_explain,
        shap_values,
        base_values,
        feature_names
    )

    # Step 10: Save summary

    save_summary(
        importance_df,
        X_explain,
        shap_values
    )

    # Final output

    print("\n")

    print("=" * 60)

    print("EARLYGUARD SHAP ANALYSIS COMPLETED!")

    print("=" * 60)

    print("\nPatient prediction:",
          "CKD" if prediction == 1 else "Not CKD")

    print(
        "CKD probability:",
        f"{probability * 100:.2f}%"
    )

    print("\nAll explainability files saved in:")

    print(EXPLAINABILITY_DIR)

    print("\nNEXT PHASE: MODEL VALIDATION AND API INTEGRATION")


# ==================================================
# 12. EXECUTION
# ==================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print("\nSHAP ANALYSIS FAILED!")

        print("Error:", error)

        raise