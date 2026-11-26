from functools import lru_cache
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "ml_artifacts" / "models" / "kidney_best_model.joblib"
PREPROCESSOR_PATH = (
    BASE_DIR / "ml_artifacts" / "preprocessing" / "kidney_preprocessor.joblib"
)

@lru_cache(maxsize=1)
def load_artifacts():
    missing = [str(path) for path in (MODEL_PATH, PREPROCESSOR_PATH) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required EarlyGuard ML artifact(s) not found: " + ", ".join(missing)
        )

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    expected_features = getattr(model, "n_features_in_", None)
    produced_features = getattr(preprocessor, "get_feature_names_out", lambda: [])()

    if expected_features is None:
        raise RuntimeError("Loaded model does not expose n_features_in_.")
    if len(produced_features) != expected_features:
        raise RuntimeError(
            f"Feature mismatch: model expects {expected_features}, "
            f"preprocessor produces {len(produced_features)}."
        )
    if list(getattr(model, "classes_", [])) != [0, 1]:
        raise RuntimeError("Expected model classes [0, 1] (Not CKD, CKD).")

    return model, preprocessor
