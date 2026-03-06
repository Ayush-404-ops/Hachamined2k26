# SmartContainer Risk Engine (Hachamined2k26)

## Overview
SmartContainer Risk Engine is an AI-powered system designed to analyze and assess the risk of shipping containers in real-time. By leveraging historical and live data from container declarations, the system predicts whether a container poses a risk—flagging it as **Critical**, **Low Risk**, or **Clear**. This application employs advanced machine learning algorithms like XGBoost and Isolation Forest to catch weight discrepancies, excessive dwell times, anomalous value-to-weight ratios, and risky behavioral patterns associated with shippers, origin countries, or HS codes.

## Key Features

The project includes the following main components:

1. **Dashboard (`dashboard.py`)**:
   A comprehensive, real-time interactive UI built with Streamlit and Plotly that features:
   - **Overview Dashboard**: High-level KPI metrics summing up the risk statistics of all processed containers, risk distribution charts, and a projected ROI calculator (hours and wages saved by avoiding random inspections).
   - **Critical Containers Drill-down**: Detailed list of high-risk shipments equipped with multi-select filters.
   - **Container Lookup**: Granular search tool allowing targeted exploration of a specific container's risk scoring, physical anomalies (e.g., measured vs. declared weight), and AI explanations behind its prediction.
   - **Geographic Risk Analysis**: A choropleth map and charts visualizing risk concentrations originating from distinct countries.
   - **Live Risk Predictor**: An on-the-fly form that computes risk scores using trained ML models for inputted container metrics.

2. **Exploratory Data Analysis (`eda.py`)**:
   A robust analytical script designed to profile container risk features, summarize missing data, test variable correlations, and plot important data distributions into `eda_plots/`. Analyzes aspects like structural weight anomalies, declared value inflation, and risky temporal behaviors (like late-night shipping declarations).

3. **Feature Engineering (`feature_engineering.py`)**:
   The data transformation pipeline. This script ingests raw container records and enriches them with new calculated indicators:
   - `weight_discrepancy` (%) and `abs_weight_discrepancy`
   - `value_per_kg` and logarithmic smoothing
   - Temporal extractions (e.g., late-night activity, weekend flags, time cyclical encodings)
   - Risk rate aggregations for specific entity profiles (e.g., Top offending Importer/Exporter IDs, HS Chapter risk rates).

4. **Modeling Pipeline (`model.py`)**:
   Trains the central Machine Learning backend:
   - **XGBoost Classifier**: Multi-class categorization to bucket risk probabilities (Clear/Low Risk/Critical) while utilizing customized class weights balancing.
   - **Isolation Forest**: Submits anomaly scores for the most extreme, atypical data points as a fail-safe layer in risk predictions.
   Leverages SHAP (SHapley Additive exPlanations) values to output human-readable, plain English explanations on *why* a shipment reached a high-risk conclusion.

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ayush-404-ops/Hachamined2k26
   cd Hachamined2k26
   ```

2. **Install Required Packages**:
   Ensure you have Python 3.8+ installed. While a `requirements.txt` file is not explicitly listed, the project requires the following primary libraries:
   ```bash
   pip install pandas numpy scikit-learn xgboost streamlit plotly seaborn matplotlib joblib shap
   ```

3. **Data Preparation**:
   The project expects real historical datasets for EDA and training.
   - `Historical Data-1.csv`: The primary raw shipping record data.
   - Scripts output interim and final files like `processed_data.csv`, `X_train.csv`, `X_test.csv`, and eventually `final_predictions.csv` which powers the dashboard predictions.
   *Note: If starting fresh, ensure the `models` folder is populated by running the training scripts or that you have pre-saved models (`xgb_model.pkl` and `isolation_forest.pkl`).*

## Running the Application

1. **Run Exploratory Data Analysis** (optional, generates plots):
   ```bash
   python eda.py
   ```

2. **Run Feature Engineering & Model Training** (optional, generates encoded datasets & `.pkl` model artifacts):
   ```bash
   python feature_engineering.py
   python model.py
   ```

3. **Launch the Dashboard**:
   ```bash
   streamlit run dashboard.py
   ```
   *The Streamlit application will become accessible through your preferred web browser.*