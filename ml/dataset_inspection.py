
from pathlib import Path
import csv

import pandas as pd


# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"


# ==================================================
# DATASET COLUMN NAMES
# ==================================================

PIMA_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age", "Outcome"
]

HEART_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "num"
]

KIDNEY_COLUMNS = [
    "age", "bp", "sg", "al", "su", "rbc",
    "pc", "pcc", "ba", "bgr", "bu", "sc",
    "sod", "pot", "hemo", "pcv", "wc", "rc",
    "htn", "dm", "cad", "appet", "pe", "ane",
    "class"
]

KIDNEY_NUMERIC = {
    "age", "bp", "bgr", "bu", "sc",
    "sod", "pot", "hemo", "pcv", "wc", "rc"
}

KIDNEY_CATEGORIES = {
    "sg": {"1.005", "1.010", "1.015", "1.020", "1.025"},
    "al": {"0", "1", "2", "3", "4", "5"},
    "su": {"0", "1", "2", "3", "4", "5"},
    "rbc": {"normal", "abnormal"},
    "pc": {"normal", "abnormal"},
    "pcc": {"present", "notpresent"},
    "ba": {"present", "notpresent"},
    "htn": {"yes", "no"},
    "dm": {"yes", "no"},
    "cad": {"yes", "no"},
    "appet": {"good", "poor"},
    "pe": {"yes", "no"},
    "ane": {"yes", "no"},
    "class": {"ckd", "notckd"}
}


# ==================================================
# DATASET INSPECTION
# ==================================================

def inspect_dataset(name, df, target=None):

    print("\n" + "=" * 65)
    print(f"DATASET: {name}")
    print("=" * 65)

    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")

    missing = df.isna().sum()

    if missing.any():
        print(missing[missing > 0].to_string())
    else:
        print("No missing values detected")

    print("\nDuplicate rows:", df.duplicated().sum())

    if target and target in df.columns:
        print(f"\nTarget distribution ({target}):")
        print(df[target].value_counts(dropna=False).to_string())

    print("\nFirst 5 records:")
    print(df.head().to_string(index=False))


# ==================================================
# KIDNEY DATA VALIDATION
# ==================================================

def valid_kidney_row(row):

    if len(row) != len(KIDNEY_COLUMNS):
        return False

    for column, value in zip(KIDNEY_COLUMNS, row):

        value = value.strip().lower()

        if value in {"", "?", "nan"}:
            continue

        if column in KIDNEY_NUMERIC:

            try:
                float(value)
            except ValueError:
                return False

        elif column in KIDNEY_CATEGORIES:

            if value not in KIDNEY_CATEGORIES[column]:
                return False

    return True


# ==================================================
# CHRONIC KIDNEY DISEASE LOADER
# ==================================================

def load_kidney_dataset():

    kidney_path = DATA_DIR / "kidney.arff"

    if not kidney_path.exists():
        raise FileNotFoundError(
            f"Kidney dataset not found: {kidney_path}"
        )

    with open(kidney_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    data_start = next(
        (
            i for i, line in enumerate(lines)
            if line.strip().lower() == "@data"
        ),
        None
    )

    if data_start is None:
        raise ValueError("Could not find @data in kidney.arff")

    records = []
    corrected_rows = 0

    for line_number, line in enumerate(
        lines[data_start + 1:],
        start=data_start + 2
    ):

        line = line.strip()

        if not line or line.startswith("%"):
            continue

        row = next(csv.reader([line], skipinitialspace=True))
        row = [value.strip().lower() for value in row]

        # Remove trailing empty fields
        while row and row[-1] == "":
            row.pop()

        # Normal 25-column record
        if len(row) == 25:

            if not valid_kidney_row(row):
                raise ValueError(
                    f"Invalid CKD values at line {line_number}:\n{row}"
                )

        # Handle one extra empty field safely
        elif len(row) == 26:

            candidates = []

            for index, value in enumerate(row):

                if value != "":
                    continue

                candidate = row[:index] + row[index + 1:]

                if valid_kidney_row(candidate):
                    candidates.append(candidate)

            if len(candidates) == 1:
                row = candidates[0]
                corrected_rows += 1

                print(
                    f"Corrected extra empty field at line {line_number}"
                )

            else:
                raise ValueError(
                    f"Cannot safely correct CKD line {line_number}.\n"
                    f"Possible corrections: {len(candidates)}\n"
                    f"Original record: {line}"
                )

        else:
            raise ValueError(
                f"CKD line {line_number}: Expected 25 fields, "
                f"found {len(row)}.\nRecord: {line}"
            )

        records.append(row)

    kidney = pd.DataFrame(records, columns=KIDNEY_COLUMNS)

    # Convert missing values
    kidney.replace(
        ["?", "", "nan"],
        pd.NA,
        inplace=True
    )

    # Convert numerical columns
    for column in KIDNEY_NUMERIC:

        kidney[column] = pd.to_numeric(
            kidney[column],
            errors="coerce"
        )

    # Print correction summary
    print("\nCKD DATA CLEANING SUMMARY")
    print("Corrected extra empty fields:", corrected_rows)
    print("Total records loaded:", len(kidney))

    return kidney


# ==================================================
# MAIN DATASET LOADING
# ==================================================

def load_and_inspect():

    # 1. Prepared Diabetes

    diabetes = pd.read_csv(
        DATA_DIR / "diabetes.csv",
        na_values=["?", "NA", ""]
    )

    inspect_dataset(
        "Diabetes - Prepared",
        diabetes,
        "Outcome" if "Outcome" in diabetes.columns else None
    )

    # 2. Original Pima Diabetes

    pima = pd.read_csv(
        DATA_DIR / "pima-indians-diabetes.data.csv",
        header=None,
        names=PIMA_COLUMNS,
        na_values=["?", "NA", ""]
    )

    inspect_dataset(
        "Pima Diabetes - Original",
        pima,
        "Outcome"
    )

    # 3. Heart Disease

    heart = pd.read_csv(
        DATA_DIR / "heart.csv",
        header=None,
        names=HEART_COLUMNS,
        na_values=["?", "NA", "", " "],
        skipinitialspace=True
    )

    inspect_dataset(
        "Heart Disease - Cleveland",
        heart,
        "num"
    )

    # 4. Chronic Kidney Disease

    kidney = load_kidney_dataset()

    # Clean categorical whitespace
    for column in kidney.select_dtypes(include="object").columns:
        kidney[column] = kidney[column].str.strip()

    inspect_dataset(
        "Chronic Kidney Disease",
        kidney,
        "class"
    )

    # 5. Diabetes comparison

    print("\n" + "=" * 65)
    print("DIABETES DATASET COMPARISON")
    print("=" * 65)

    print("Prepared Diabetes:", diabetes.shape)
    print("Original Pima:", pima.shape)

    print("Prepared columns:", list(diabetes.columns))
    print("Original columns:", list(pima.columns))

    # 6. Final summary

    print("\n" + "=" * 65)
    print("EARLYGUARD DATASET INSPECTION COMPLETED")
    print("=" * 65)

    print("Diabetes:", diabetes.shape)
    print("Pima Diabetes:", pima.shape)
    print("Heart Disease:", heart.shape)
    print("Kidney Disease:", kidney.shape)


# ==================================================
# EXECUTION
# ==================================================

if __name__ == "__main__":
    load_and_inspect()
