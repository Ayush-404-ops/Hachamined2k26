# Integrated SmartContainer Risk Engine

This project integrates a beautiful, high-performance React frontend from **Sentinel-AI** with the powerful Machine Learning backend from **Hachamined2k26**.

## 🚀 Architecture

- **Backend**: FastAPI (Python 3.10+) serving XGBoost and Isolation Forest models.
  - Port: `8000`
  - Swagger Documentation: `http://localhost:8000/docs`
- **Frontend**: React + Vite + Tailwind CSS (Shadcn UI).
  - Port: `8081`
- **Machine Learning**: 
  - `xgb_model.pkl`: Classification (Risk Levels)
  - `isolation_forest.pkl`: Anomaly Detection

## 🛠️ Getting Started

### 1. Prerequisites
- Python 3.10+
- Node.js & npm

### 2. Setup Backend
1. Install dependencies:
   ```bash
   pip install fastapi uvicorn pandas numpy joblib scikit-learn
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
   (The server will usually start on `http://localhost:8081`)

## 📊 Features

- **Overview Dashboard**: Real-time KPI aggregation and trend analysis across 54,000+ records.
- **Critical Alerts**: Filterable list of high-risk containers flagged by the ML models.
- **Geographic Risk**: Interactive visualization of global risk distribution.
- **Live Predictor**: Direct ML model polling for real-time shipment analysis.
- **Container Lookup**: Detailed SHAP value explanation for every container ID.
- **System Settings**: Extensive configuration panel for general preferences, risk thresholds, API backend, notifications, and UI appearance.