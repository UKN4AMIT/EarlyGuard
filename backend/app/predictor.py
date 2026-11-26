import numpy as np
import pandas as pd

from app.model_loader import load_artifacts

RAW_COLUMNS = [
    "age", "bp", "sg", "al", "su", "rbc",
    "pc", "pcc", "ba", "bgr", "bu", "sc",
    "sod", "pot", "hemo", "pcv", "wc", "rc",
    "htn", "dm", "cad", "appet", "pe", "ane",
]

def predict_kidney_disease(patient_data: dict) -> dict:
    model, preprocessor = load_artifacts()

    # Keep the exact raw feature order used during training.
    patient_df = pd.DataFrame(
        [[patient_data.get(column) for column in RAW_COLUMNS]],
        columns=RAW_COLUMNS,
    )

    # The fitted training preprocessor handles missing values and encoding.
    transformed = np.asarray(preprocessor.transform(patient_df), dtype=float)

    if transformed.shape[1] != model.n_features_in_:
        raise RuntimeError(
            f"Model expects {model.n_features_in_} features, "
            f"received {transformed.shape[1]}."
        )
    if not np.isfinite(transformed).all():
        raise RuntimeError("Preprocessing produced non-finite values.")

    probabilities = model.predict_proba(transformed)[0]
    classes = list(model.classes_)
    ckd_probability = float(probabilities[classes.index(1)])
    not_ckd_probability = float(probabilities[classes.index(0)])
    predicted_class = int(model.predict(transformed)[0])

    return {
        "prediction": "CKD" if predicted_class == 1 else "Not CKD",
        "predicted_class": predicted_class,
        "ckd_probability": round(ckd_probability, 6),
        "not_ckd_probability": round(not_ckd_probability, 6),
        "model": "Logistic Regression",
        "disclaimer": (
            "Educational research prototype only. This output is not a diagnosis "
            "or a substitute for evaluation by a qualified healthcare professional."
        ),
    }
