# EarlyGuard Backend (ML-integrated)

This backend connects the existing CKD preprocessing artifact and Logistic Regression model to FastAPI.

## Folder layout

- `app/`: FastAPI application and prediction code
- `ml_artifacts/preprocessing/kidney_preprocessor.joblib`: fitted 24-column preprocessor
- `ml_artifacts/models/kidney_best_model.joblib`: trained 34-feature classifier
- `tests/`: API tests

## Important compatibility note

The supplied model artifacts were serialized with scikit-learn 1.9.1. The requirements file pins that version. Use the same Python environment/version family used to create the artifacts where possible. Do not load untrusted `.joblib` files.

## Setup (run from the backend folder)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`.

## Endpoints

- `GET /`
- `GET /api/health`
- `GET /api/model/status`
- `POST /api/predict/kidney`

The prediction endpoint accepts the 24 raw input columns used by the training preprocessor. Fields can be omitted or set to `null`; the fitted imputer handles missing values. At least one value must be supplied.

## Test

```bash
pytest -q
```

## Clinical safety

This is an educational/research prototype. It must not be used for diagnosis, screening, or treatment decisions. The displayed probability is a model output, not a clinically calibrated individual risk estimate. Validate externally before any real-world use.
