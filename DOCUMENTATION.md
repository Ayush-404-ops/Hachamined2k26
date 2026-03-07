# SmartContainer Risk Engine - Comprehensive Documentation

## Overview
The **SmartContainer Risk Engine** is a state-of-the-art port security platform designed to integrate powerful Machine Learning with a highly responsive, glassmorphic React user interface. 

The system leverages advanced tree-based classification and anomaly detection to flag suspicious shipping containers based on historic intelligence (54,000+ real-world port records), minimizing smuggling and customs fraud seamlessly in real time.

---

## 1. System Architecture

### Frontend (User Interface)
- **Framework**: React 18 with Vite
- **Styling**: Tailwind CSS, Shadcn UI components, Framer Motion for animations
- **3D Visualization**: React Three Fiber / Drei (used for the 3D container lookup view)
- **Design System**: Premium dark-mode aesthetic (#0D1117 backgrounds) with dynamic blue and red accents to signify risk levels.
- **Key Modules**:
  - `Index.tsx`: The main dashboard overview with KPI aggregation.
  - `CriticalAlerts.tsx`: Filterable list of all containers flagged by the ML pipeline.
  - `LivePredictor.tsx`: A live data feed directly tied to the FastAPI backend polling.
  - `ContainerLookup.tsx`: Interactive search with visual 3D representation and SHAP value context.
  - `SettingsPage.tsx`: Extensible module for system thresholds, profile settings, and API management.

### Backend (Machine Learning Pipeline)
- **Framework**: FastAPI (Python 3.10+)
- **Models Used**:
  1. **XGBoost Classifier (`xgb_model.pkl`)**: Predicts the primary probability that a container is carrying undocumented/illicit cargo based on historical patterns (origin, weight discrepancy, HS codes).
  2. **Isolation Forest (`isolation_forest.pkl`)**: Provides an overlapping anomaly score detecting entirely new, "zero-day" statistical shifts in smuggling methodologies.
- **Supporting Scripts**:
  - `api.py`: Hosts the REST endpoints.
  - `model.py` & `feature_engineering.py`: Data processing pipeline and model training loops.
  - `dashboard.py`: Legacy Streamlit dashboard module initially used for explorative analysis.

---

## 2. API Endpoints (`localhost:8000`)

To view interactive Swagger docs, visit `http://localhost:8000/docs`.

### Key Endpoints:
- `GET /api/v1/health`: Returns basic model health status, showing if XGBoost and Isolation models are correctly loaded into memory.
- `GET /api/v1/containers/critical`: Returns a paginated list of all currently flagged containers meeting the "critical" threshold.
- `POST /api/v1/predict`: Accepts JSON payload of manifesting container data, returning immediate probability distributions for risk level.
- `GET /api/v1/lookup/{container_id}`: Retrieves deep context (including SHAP/feature importance data) for a specific ID.

---

## 3. Configuration & Risk Thresholds

The frontend UI incorporates an interactive **Settings** panel mapped to system thresholds:
- **Critical Risk Level (Default: >70%)**: Flagged in striking red; triggers alerts and lockdown signals.
- **Low-Risk Warning (Default: >35%)**: Flagged in amber; signals secondary inspection without stalling primary terminal operations.
- **Weight Discrepancy Margin (Default: 20%)**: Variance allowed between the declared manifest weight and sensors prior to raising flags.

These thresholds can be hot-swapped by terminal operators directly via the `SettingsPage.tsx` interface. 

---

## 4. Setup & Deployment Instructions

### Local Development

**1. Start the Machine Learning API**
```bash
cd /path/to/hackamined2.0
pip install fastapi uvicorn pandas numpy scikit-learn joblib xgboost
uvicorn api:app --reload --port 8000
```

**2. Start the Frontend Application**
```bash
cd /path/to/hackamined2.0/sentinel-ui
npm install
npm run dev -- --port 8081
```

Once running:
- Open **http://localhost:8081** for the web interface
- Open **http://localhost:8000/docs** for the API specifications

### Deployment Guidelines
- **Backend Deployment**: Containerize the FastAPI application using Docker. Ensure the `models/` folder containing the `.pkl` files is included via build or mounted as a volume. Deploy via AWS ECS, GCP Cloud Run, or generic managed Kubernetes.
- **Frontend Deployment**: Build using `npm run build`. This packages the Vite app strictly into a static `dist/` folder. This can be securely hosted anywhere edge-optimized (Vercel, AWS CloudFront + S3, Netlify).

---

## 5. Security & Maintenance

- **API Security**: In production environments, place the FastAPI backend behind an API Gateway to enforce JWT-based authorization and rate-limiting.
- **Model Retraining**: Current models are snapshotted in the `/models` directory. To avoid model drift as customs data matures, data engineers should periodically run `model.py` pointed towards updated `X_train.csv` and `y_train.csv` datasets.
