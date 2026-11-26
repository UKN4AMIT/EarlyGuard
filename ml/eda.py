
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "artifacts" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# ==========================================
# DATASET LOADERS
# ==========================================

def load_datasets():

    datasets = {}

    # Diabetes
    datasets["Diabetes"] = pd.read_csv(
        DATA_DIR / "diabetes.csv"
    )

    # Heart Disease
    heart_columns = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal", "num"
    ]

    datasets["Heart Disease"] = pd.read_csv(
        DATA_DIR / "heart.csv",
        header=None,
        names=heart_columns,
        na_values=["?", "NA", ""],
        skipinitialspace=True
    )

    # Chronic Kidney Disease
    kidney_path = DATA_DIR / "kidney.arff"

    lines = kidney_path.read_text(
        encoding="utf-8"
    ).splitlines()

    data_start = next(
        i for i, line in enumerate(lines)
        if line.strip().lower() == "@data"
    )

    kidney_columns = [
        "age", "bp", "sg", "al", "su", "rbc",
        "pc", "pcc", "ba", "bgr", "bu", "sc",
        "sod", "pot", "hemo", "pcv", "wc", "rc",
        "htn", "dm", "cad", "appet", "pe", "ane",
        "class"
    ]

    from io import StringIO
    import csv

    records = []

    for line in lines[data_start + 1:]:

        line = line.strip()

        if not line or line.startswith("%"):
            continue

        row = next(csv.reader([line], skipinitialspace=True))

        row = [value.strip() for value in row]

        while row and row[-1] == "":
            row.pop()

        
        # Validate and handle record length
        if len(row) > len(kidney_columns):
            row = row[:len(kidney_columns)]

        if len(row) < len(kidney_columns):
            raise ValueError(
                f"Invalid kidney record: {line}"
            )

        records.append(row)

    kidney = pd.DataFrame(
        records,
        columns=kidney_columns
    )

    kidney.replace(["?", ""], np.nan, inplace=True)

    for col in kidney.columns:
        kidney[col] = pd.to_numeric(
            kidney[col], errors="ignore"
        ) if col not in kidney.select_dtypes(
            include="object"
        ).columns else kidney[col]

    for col in kidney.columns:
        if col != "class":
            kidney[col] = pd.to_numeric(
                kidney[col], errors="ignore"
            )

    datasets["Kidney Disease"] = kidney

    return datasets


# ==========================================
# DATASET EDA
# ==========================================

def perform_eda(name, df):

    print("\n" + "=" * 60)
    print(f"EDA: {name}")
    print("=" * 60)

    print("Shape:", df.shape)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    # Save missing value chart
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if not missing.empty:

        plt.figure(figsize=(10, 5))
        missing.sort_values().plot(kind="bar")

        plt.title(f"{name} - Missing Values")
        plt.ylabel("Count")
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_missing.png"
        )
        plt.close()

    # Target distribution
    target = next(
        (col for col in ["Outcome", "num", "class"]
         if col in df.columns),
        None
    )

    if target:

        plt.figure(figsize=(7, 5))

        sns.countplot(
            data=df,
            x=target
        )

        plt.title(f"{name} - Target Distribution")
        plt.xticks(rotation=30)
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_target.png"
        )
        plt.close()

    # Numerical feature histograms
    numeric_df = df.select_dtypes(include=np.number)

    if not numeric_df.empty:

        numeric_df.hist(
            figsize=(14, 10),
            bins=20
        )

        plt.suptitle(f"{name} - Feature Distributions")
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_distributions.png"
        )
        plt.close()

        # Correlation heatmap
        plt.figure(figsize=(12, 9))

        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f"
        )

        plt.title(f"{name} - Correlation Heatmap")
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_correlation.png"
        )
        plt.close()

    print("EDA completed for:", name)


# ==========================================
# MAIN
# ==========================================

def main():

    datasets = load_datasets()

    for name, df in datasets.items():
        perform_eda(name, df)

    print("\n" + "=" * 60)
    print("EARLYGUARD EDA COMPLETED")
    print("Graphs saved at:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
