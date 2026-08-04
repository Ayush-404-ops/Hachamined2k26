# SmartContainer Risk Engine - Resume Notes

## Project Snapshot

- Built an AI-powered customs and port-risk platform that scores shipping containers for smuggling, fraud, and anomaly risk.
- Processed 54,000 historical port records with 53 engineered model features.
- Combined an XGBoost risk classifier with an Isolation Forest anomaly detector.
- Served model/data insights through a FastAPI backend and a React + Vite + Tailwind dashboard.
- Added interactive workflows for KPI monitoring, critical alerts, geographic risk analysis, live prediction, container lookup, and model-performance reports.

## Resume Bullets

- Developed a full-stack SmartContainer Risk Engine using FastAPI, React, TypeScript, XGBoost, and Isolation Forest to classify shipment risk across 54,000 historical port records.
- Engineered and integrated REST APIs for live prediction, critical-container filtering, geographic risk analytics, model health, ROI metrics, and model-performance reporting.
- Improved frontend maintainability by adding shared TypeScript API contracts, replacing loose `any` usage, and making `npm run lint` pass cleanly.
- Added backend health/versioned API compatibility routes and aligned documentation with the actual FastAPI surface.
- Verified production readiness with Python import checks, backend model/data loading, Vitest, ESLint, and Vite production build.

## Tech Stack

- Frontend: React 18, Vite, TypeScript, Tailwind CSS, shadcn UI, Recharts, Framer Motion, React Three Fiber
- Backend: Python, FastAPI, Pydantic, Pandas, NumPy, Joblib
- ML: XGBoost classifier, Isolation Forest anomaly detection, feature engineering pipeline
- Data/Artifacts: 54,000 prediction records, processed training/test CSVs, model pickles, feature-name schema, EDA/model plots

## Short Description

SmartContainer Risk Engine is a full-stack AI risk-intelligence dashboard for customs teams, combining machine-learning shipment scoring, anomaly detection, visual analytics, and live prediction APIs to prioritize high-risk containers for inspection.
