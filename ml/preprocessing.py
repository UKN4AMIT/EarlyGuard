
from pathlib import Path
import csv
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "raw"
KIDNEY_PATH = DATA_DIR / "kidney.arff"

ARTIFACT_DIR = BASE_DIR / "artifacts" / "preprocessing"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. DATASET COLUMNS
# ==================================================

KIDNEY_COLUMNS = [
    "age", "bp", "sg", "al", "su", "rbc",
    "pc", "pcc", "ba", "bgr", "bu", "sc",
    "sod", "pot", "hemo", "pcv", "wc", "rc",
    "htn", "dm", "cad", "appet", "pe", "ane",
    "class"
]

NUMERIC_COLUMNS = [
    "age", "bp", "sg", "al", "su", "bgr",
    "bu", "sc", "sod", "pot", "hemo",
    "pcv", "wc", "rc"
]

CATEGORICAL_COLUMNS = [
    "rbc", "pc", "pcc", "ba", "htn",
    "dm", "cad", "appet", "pe", "ane"
]

TARGET_COLUMN = "class"


# ==================================================
# 3. LOAD AND CLEAN KIDNEY DATASET
# ==================================================

def load_kidney_dataset():

    print("\nLoading Kidney Disease Dataset...")

    if not KIDNEY_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {KIDNEY_PATH}"
        )

    with open(
        KIDNEY_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        lines = file.readlines()

    # Locate ARFF data section
    data_start = next(
        (
            i for i, line in enumerate(lines)
            if line.strip().lower() == "@data"
        ),
        None
    )

    if data_start is None:
        raise ValueError(
            "Could not find @data section in kidney.arff"
        )

    records = []
    corrected_records = 0
    skipped_records = 0

    # Read dataset records
    for line_number, line in enumerate(
        lines[data_start + 1:],
        start=data_start + 2
    ):

        line = line.strip()

        if not line or line.startswith("%"):
            continue

        row = next(
            csv.reader(
                [line],
                skipinitialspace=True
            )
        )

        row = [
            value.strip()
            for value in row
        ]

        # Remove unnecessary trailing empty fields
        while row and row[-1] == "":
            row.pop()

        # Handle the known extra-empty-field record
        if len(row) == len(KIDNEY_COLUMNS) + 1:

            empty_positions = [
                i for i, value in enumerate(row)
                if value == ""
            ]

            if len(empty_positions) == 1:

                del row[empty_positions[0]]
                corrected_records += 1

            else:
                raise ValueError(
                    f"Unexpected malformed record at line "
                    f"{line_number}: {row}"
                )

        # Validate number of fields
        if len(row) != len(KIDNEY_COLUMNS):

            raise ValueError(
                f"Invalid record at line {line_number}. "
                f"Expected {len(KIDNEY_COLUMNS)} fields, "
                f"found {len(row)}."
            )

        records.append(row)

    # Create DataFrame
    kidney = pd.DataFrame(
        records,
        columns=KIDNEY_COLUMNS
    )

    # Standardize missing values
    kidney.replace(
        ["?", "", " ?"],
        np.nan,
        inplace=True
    )

    # Clean numerical columns
    for col in NUMERIC_COLUMNS:

        kidney[col] = pd.to_numeric(
            kidney[col],
            errors="coerce"
        )

    # Clean categorical columns
    # Use np.nan instead of pandas pd.NA
    for col in CATEGORICAL_COLUMNS + [TARGET_COLUMN]:

        kidney[col] = kidney[col].map(
            lambda value: (
                value.strip().lower()
                if isinstance(value, str)
                else np.nan
            )
        ).astype(object)

    # Remove records without target labels
    kidney.dropna(
        subset=[TARGET_COLUMN],
        inplace=True
    )

    # Validate target labels
    valid_targets = {"ckd", "notckd"}

    unexpected = (
        set(kidney[TARGET_COLUMN].dropna().unique())
        - valid_targets
    )

    if unexpected:
        raise ValueError(
            f"Unexpected target labels: {unexpected}"
        )

    print("\nDataset loading completed.")
    print("Valid records:", len(kidney))
    print("Corrected malformed records:", corrected_records)
    print("Skipped malformed records:", skipped_records)

    return kidney


# ==================================================
# 4. PREPROCESSING FUNCTION
# ==================================================

def preprocess_kidney_dataset():

    print("\n" + "=" * 60)
    print("EARLYGUARD - CKD DATA PREPROCESSING")
    print("=" * 60)

    # Load dataset
    df = load_kidney_dataset()

    print("\nOriginal dataset shape:", df.shape)

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])

    y = df[TARGET_COLUMN].map({
        "ckd": 1,
        "notckd": 0
    }).astype(int)

    # Ensure categorical columns use ordinary Python objects
    # and missing values use np.nan
    for col in CATEGORICAL_COLUMNS:

        X[col] = X[col].astype(object)

        X[col] = X[col].where(
            pd.notna(X[col]),
            np.nan
        )

    # Target distribution
    print("\nTarget distribution:")
    print(
        y.map({
            0: "Not CKD",
            1: "CKD"
        }).value_counts()
    )

    # ==============================================
    # TRAIN TEST SPLIT
    # ==============================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # ==============================================
    # NUMERICAL PIPELINE
    # ==============================================

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # ==============================================
    # CATEGORICAL PIPELINE
    # ==============================================

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # ==============================================
    # COMBINE BOTH PIPELINES
    # ==============================================

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_COLUMNS
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_COLUMNS
            )
        ],
        remainder="drop"
    )

    # ==============================================
    # FIT ON TRAINING DATA ONLY
    # ==============================================

    print("\nFitting preprocessing pipeline...")

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    # Transform test data
    X_test_processed = preprocessor.transform(
        X_test
    )

    # Convert to NumPy arrays
    X_train_processed = np.asarray(
        X_train_processed,
        dtype=np.float64
    )

    X_test_processed = np.asarray(
        X_test_processed,
        dtype=np.float64
    )

    # ==============================================
    # VALIDATION
    # ==============================================

    if not np.isfinite(X_train_processed).all():
        raise ValueError(
            "Invalid values found in training data."
        )

    if not np.isfinite(X_test_processed).all():
        raise ValueError(
            "Invalid values found in testing data."
        )

    # ==============================================
    # SAVE PREPROCESSOR AND DATA
    # ==============================================

    joblib.dump(
        preprocessor,
        ARTIFACT_DIR / "kidney_preprocessor.joblib"
    )

    np.savez_compressed(
        ARTIFACT_DIR / "kidney_processed_data.npz",
        X_train=X_train_processed,
        X_test=X_test_processed,
        y_train=y_train.to_numpy(),
        y_test=y_test.to_numpy()
    )

    # ==============================================
    # FINAL RESULTS
    # ==============================================

    print("\n" + "=" * 60)
    print("PREPROCESSING RESULTS")
    print("=" * 60)

    print(
        "Training data shape:",
        X_train_processed.shape
    )

    print(
        "Testing data shape:",
        X_test_processed.shape
    )

    print(
        "Training missing/invalid values:",
        np.size(X_train_processed)
        - np.isfinite(X_train_processed).sum()
    )

    print(
        "Testing missing/invalid values:",
        np.size(X_test_processed)
        - np.isfinite(X_test_processed).sum()
    )

    print("\nSaved files:")

    print(
        ARTIFACT_DIR / "kidney_preprocessor.joblib"
    )

    print(
        ARTIFACT_DIR / "kidney_processed_data.npz"
    )

    print("\n" + "=" * 60)
    print("EARLYGUARD PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )


# ==================================================
# 5. MAIN EXECUTION
# ==================================================

if __name__ == "__main__":

    try:

        (
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor
        ) = preprocess_kidney_dataset()

    except Exception as error:

        print("\nPREPROCESSING FAILED!")
        print("Error:", error)

        raise
