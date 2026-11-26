from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.model_loader import load_artifacts
from app.routes.prediction import router as prediction_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail early if model files are absent or incompatible.
    load_artifacts()
    yield

app = FastAPI(
    title="EarlyGuard API",
    description=(
        "Educational explainable-AI chronic kidney disease prediction prototype. "
        "Not for clinical diagnosis or treatment decisions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Development CORS. Replace with your deployed frontend origin before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction_router)

@app.get("/", tags=["system"])
def root():
    return {"message": "EarlyGuard API is running", "docs": "/docs"}

@app.get("/api/health", tags=["system"])
def health():
    return {
        "status": "ok",
        "service": "earlyguard-api",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/api/model/status", tags=["system"])
def model_status():
    model, preprocessor = load_artifacts()
    return {
        "loaded": True,
        "model": type(model).__name__,
        "model_features": int(model.n_features_in_),
        "preprocessor_output_features": len(preprocessor.get_feature_names_out()),
        "classes": [int(value) for value in model.classes_],
    }
