# EarlyGuard
Explainable AI-Based Early Disease Risk Prediction System.

## Phase 0: local development bootstrap
This repository starts with a lightweight local setup for macOS (Apple Silicon).

### Prerequisites
- Python 3.11 or 3.12
- Node.js 20 LTS or newer
- VS Code

### Backend
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs and http://127.0.0.1:8000/api/health.

### Frontend
Open a second terminal:
```bash
cd frontend
npm install
npm run dev
```
Open the URL printed by Vite (normally http://localhost:5173).

### Next phases
1. Inspect and document the selected public datasets.
2. Build leakage-safe training pipelines and evaluate real models.
3. Add SHAP explanations.
4. Connect trained model artifacts to FastAPI.
5. Add authentication, private assessment history, reports, tests and deployment.

No clinical performance claims or fabricated metrics are included. This is an educational prototype, not a diagnostic device.
