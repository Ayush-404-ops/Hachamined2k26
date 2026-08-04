# Integrated SmartContainer Risk Engine

This project integrates a high-performance React frontend from **Sentinel-AI** with the Machine Learning backend from **Hachamined2k26**.

## Architecture

- **Backend**: FastAPI (Python 3.10+) serving XGBoost and Isolation Forest models.
  - Port: `8000`
  - Swagger documentation: `http://localhost:8000/docs`
- **Frontend**: React + Vite + Tailwind CSS + shadcn UI.
  - Port: `8081`
- **Machine Learning**:
  - `xgb_model.pkl`: classification model for container risk levels.
  - `isolation_forest.pkl`: anomaly detection model.

## Getting Started

### 1. Prerequisites

- Python 3.10+
- Node.js and npm

### 2. Setup Backend

1. Install dependencies:

   ```bash
   pip install fastapi uvicorn pandas numpy joblib scikit-learn xgboost
   ```

2. Start the FastAPI server:

   ```bash
   uvicorn api:app --reload --port 8000
   ```

### 3. Setup Frontend

1. Navigate to the `sentinel-ui` directory:

   ```bash
   cd sentinel-ui
   ```

2. Install dependencies:

   ```bash
   npm install
   ```

3. Start the development server:

   ```bash
   npm run dev -- --port 8081
   ```

The frontend will usually start on `http://localhost:8081`.

## API Quick Check

- Health: `GET http://localhost:8000/api/health`
- Swagger docs: `http://localhost:8000/docs`
- Main API prefix used by the frontend: `/api`
- Versioned compatibility aliases: `/api/v1/health`, `/api/v1/containers/critical`, `/api/v1/predict`, and `/api/v1/lookup/{container_id}`

## Features

- **Overview Dashboard**: Real-time KPI aggregation and trend analysis across 54,000+ records.
- **Critical Alerts**: Filterable list of high-risk containers flagged by the ML models.
- **Geographic Risk**: Interactive visualization of global risk distribution.
- **Live Predictor**: Direct ML model polling for real-time shipment analysis.
- **Container Lookup**: Detailed model explanation context for every container ID.
- **System Settings**: Configuration panel for preferences, thresholds, API backend, notifications, and UI appearance.
