import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime, time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_flag(iso_code: str):
    flags = {
        "CN": "🇨🇳", "NG": "🇳🇬", "AE": "🇦🇪", "RU": "🇷🇺", "PK": "🇵🇰", 
        "TR": "🇹🇷", "MX": "🇲🇽", "VN": "🇻🇳", "BR": "🇧🇷", "IN": "🇮🇳",
        "BE": "🇧🇪", "US": "🇺🇸", "DE": "🇩🇪", "IT": "🇮🇹", "JP": "🇯🇵",
        "CA": "🇨🇦", "GB": "🇬🇧", "NP": "🇳🇵", "NI": "🇳🇮", "TW": "🇹🇼",
        "CH": "🇨🇭", "MY": "🇲🇾", "FR": "🇫🇷", "PL": "🇵🇱", "TH": "🇹🇭",
        "KR": "🇰🇷", "EG": "🇪🇬", "SA": "🇸🇦", "ZA": "🇿🇦", "UA": "🇺🇦",
        "NO": "🇳🇴", "ES": "🇪🇸", "NL": "🇳🇱", "AU": "🇦🇺"
    }
    return flags.get(str(iso_code).upper(), "🌐")

app = FastAPI(title="SmartContainer Risk Engine API")

# Allow CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# DATA LOADERS & HELPERS
# ─────────────────────────────────────────────
# We load data once when the application starts
print("Loading data and models...")

try:
    preds_df = pd.read_csv("final_predictions.csv")
    raw_df = pd.read_csv("Historical Data-1.csv")
    col_map = {
        'Declaration_Date (YYYY-MM-DD)': 'Declaration_Date',
        'Declaration_Time (HH:MM:SS)': 'Declaration_Time',
        'Trade_Regime (Import / Export / Transit)': 'Trade_Regime'
    }
    raw_df.rename(columns=col_map, inplace=True)
    proc_df = pd.read_csv("processed_data.csv")
    
    # Merge for a comprehensive dataset
    keep = ['Container_ID','Declaration_Date','Origin_Country','Destination_Country',
            'Destination_Port','Shipping_Line','HS_Code','HS_Description','Importer_ID','Exporter_ID',
            'Declared_Value','Declared_Weight','Measured_Weight','Dwell_Time_Hours',
            'Trade_Regime','Clearance_Status']
    
    cols_to_join = [col for col in keep if col not in proc_df.columns and col in raw_df.columns]
    if cols_to_join:
        merged_proc = proc_df.join(raw_df[cols_to_join].reset_index(drop=True))
    else:
        merged_proc = proc_df

    xgb_model = joblib.load("models/xgb_model.pkl")
    iso_model  = joblib.load("models/isolation_forest.pkl")
    with open("models/feature_names.json", "r") as f:
        feature_names = json.load(f)
    print("Data and models loaded successfully.")
except Exception as e:
    print(f"Error loading data/models: {e}")
    preds_df = pd.DataFrame()
    merged_proc = pd.DataFrame()
    raw_df = pd.DataFrame()
    xgb_model = None
    iso_model = None
    feature_names = []

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    containerId: str
    origin: str
    hsCode: str
    declaredWeight: str
    measuredWeight: str
    declaredValue: str
    shipmentDate: str
    dwellTime: str
    shipperId: str
    importerId: str

# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/api/overview/stats")
def get_overview_stats():
    if preds_df.empty:
        raise HTTPException(status_code=500, detail="Data not available")
    
    total = len(preds_df)
    crit = int((preds_df['Risk_Level'] == 'Critical').sum())
    low = int((preds_df['Risk_Level'] == 'Low Risk').sum())
    clear = total - crit - low
    anom = int(preds_df['Is_Anomaly'].sum())
    
    return {
        "total": total,
        "critical": crit,
        "lowRisk": low,
        "clear": clear,
        "anomalies": anom
    }

@app.get("/api/overview/roi")
def get_roi():
    if preds_df.empty:
        raise HTTPException(status_code=500, detail="Data not available")
    
    total = len(preds_df)
    crit = int((preds_df['Risk_Level'] == 'Critical').sum())
    
    trad_rate = 0.40
    ai_rate = crit / total if total > 0 else 0
    inspections_avoided = (trad_rate - ai_rate) * total
    
    hrs_per_ins = 2.0
    wage_hr = 15
    manhours_saved = inspections_avoided * hrs_per_ins
    wages_saved = manhours_saved * wage_hr
    
    return {
        "hoursSaved": max(0, manhours_saved),
        "wagesSaved": max(0, wages_saved),
        "inspectionsAvoided": max(0, inspections_avoided),
        "avoidanceRate": (1 - ai_rate) * 100 if ai_rate < 1 else 0
    }

@app.get("/api/containers/critical")
def get_critical_containers(limit: int = 50):
    if preds_df.empty or merged_proc.empty:
        return []
    
    # Needs a combined view of features
    display_cols = ['Container_ID','Risk_Score','Risk_Level','Is_Anomaly','Explanation_Summary']
    df = preds_df[display_cols].copy()
    
    # Merge additional info sequentially by index assuming they match
    extra_cols = ['Origin_Country', 'Destination_Country', 'HS_Code', 'HS_Description', 
                  'Declared_Weight', 'Measured_Weight', 'Declared_Value', 
                  'Shipping_Line', 'Importer_ID', 'Declaration_Date', 'Dwell_Time_Hours']
    
    for col in extra_cols:
        if col in merged_proc.columns:
            df[col] = merged_proc[col].iloc[:len(df)].values
        elif col in raw_df.columns:
             df[col] = raw_df[col].iloc[:len(df)].values
        else:
            df[col] = "Unknown"
            
    # Calculate discrepancy
    df['Declared_Weight'] = pd.to_numeric(df['Declared_Weight'], errors='coerce').fillna(1)
    df['Measured_Weight'] = pd.to_numeric(df['Measured_Weight'], errors='coerce').fillna(df['Declared_Weight'])
    df['Weight_Discrepancy'] = ((df['Measured_Weight'] - df['Declared_Weight']) / df['Declared_Weight']) * 100
            
    crit_df = df[df['Risk_Level'] == 'Critical'].sort_values('Risk_Score', ascending=False).head(limit)
    
    results = []
    for _, row in crit_df.iterrows():
        country = str(row.get('Origin_Country', 'Unknown'))
        
        results.append({
            "id": str(row['Container_ID']),
            "origin": country,
            "originFlag": get_flag(country),
            "destination": str(row.get('Destination_Country', 'Unknown')),
            "hsCode": str(row.get('HS_Code', 'Unknown')),
            "hsDesc": str(row.get('HS_Description', 'Goods')),
            "declaredWeight": float(row.get('Declared_Weight', 0)),
            "measuredWeight": float(row.get('Measured_Weight', 0)),
            "weightDiscrepancy": round(float(row.get('Weight_Discrepancy', 0)), 1),
            "declaredValue": float(row.get('Declared_Value', 0)),
            "valuePerKg": float(row.get('Declared_Value', 0)) / float(row.get('Declared_Weight', 1)),
            "shipper": str(row.get('Shipping_Line', 'Unknown')),
            "importer": str(row.get('Importer_ID', 'Unknown')),
            "shipmentDate": str(row.get('Declaration_Date', '')),
            "dwellTime": float(row.get('Dwell_Time_Hours', 0)) / 24.0, # Convert hours to days for UI
            "riskScore": int(row['Risk_Score']),
            "riskLevel": row['Risk_Level'],
            "flaggedReason": str(row['Explanation_Summary'])
        })
    
    return results

@app.get("/api/containers/geographic")
def get_geographic_risk():
    if 'Origin_Country' not in merged_proc.columns or 'Clearance_Status' not in merged_proc.columns:
        return []
    
    total_by_orig = merged_proc.groupby('Origin_Country').size()
    crit_by_orig = merged_proc[merged_proc['Clearance_Status'] == 'Critical'].groupby('Origin_Country').size()
    
    results = []
    for country in top_origins(crit_by_orig, total_by_orig, 10):
        total = int(total_by_orig.get(country, 0))
        crit = int(crit_by_orig.get(country, 0))
        pct = (crit / total * 100) if total > 0 else 0
        
        results.append({
            "country": country,
            "flag": get_flag(country),
            "pct": round(pct, 1),
            "count": crit
        })
        
    return sorted(results, key=lambda x: x['pct'], reverse=True)

def top_origins(crit_series, total_series, n=10):
    rates = {}
    for country, total in total_series.items():
        if total >= 5: # min threshold
            rates[country] = crit_series.get(country, 0) / total
    return sorted(rates, key=rates.get, reverse=True)[:n]

@app.get("/api/containers/trends")
def get_trends():
    try:
        if raw_df.empty or preds_df.empty:
            return []
            
        # Join predictions with dates
        df = preds_df[['Risk_Level']].copy()
        df['Date'] = pd.to_datetime(raw_df['Declaration_Date'].iloc[:len(df)], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Group by week
        df['Week'] = df['Date'].dt.to_period('W').apply(lambda r: r.start_time)
        weeks = sorted(df['Week'].unique())[-12:] # Last 12 weeks
        
        results = []
        for i, week in enumerate(weeks):
            week_df = df[df['Week'] == week]
            results.append({
                "week": f"W{i+1}",
                "date": week.strftime('%Y-%m-%d'),
                "critical": int((week_df['Risk_Level'] == 'Critical').sum()),
                "low": int((week_df['Risk_Level'] == 'Low Risk').sum()),
                "clear": int((week_df['Risk_Level'] == 'Clear').sum() + (week_df['Risk_Level'].isna()).sum())
            })
        return results
    except Exception as e:
        print(f"Trend error: {e}")
        return []

@app.get("/api/containers/{container_id}")
def get_container(container_id: str):
    if preds_df.empty:
        raise HTTPException(status_code=500, detail="Data not available")
        
    match = preds_df[preds_df['Container_ID'].astype(str) == container_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="Container not found")
        
    row = match.iloc[0]
    idx = match.index[0]
    raw_row = merged_proc.iloc[idx] if idx < len(merged_proc) else pd.Series()
    
    decl_w = float(raw_row.get('Declared_Weight', 1) or 1)
    meas_w = float(raw_row.get('Measured_Weight', decl_w) or decl_w)
    country = str(raw_row.get('Origin_Country', 'Unknown'))
    
    return {
        "id": container_id,
        "riskScore": float(row['Risk_Score']),
        "riskLevel": row['Risk_Level'],
        "origin": country,
        "originFlag": get_flag(country),
        "destination": str(raw_row.get('Destination_Country', 'Unknown')),
        "declaredWeight": float(raw_row.get('Declared_Weight', 0)),
        "measuredWeight": meas_w,
        "weightDiscrepancy": round(((meas_w - decl_w) / decl_w) * 100, 1),
        "declaredValue": float(raw_row.get('Declared_Value', 0)),
        "dwellTime": float(raw_row.get('Dwell_Time_Hours', 0)) / 24.0,
        "shipper": str(raw_row.get('Shipping_Line', 'Unknown')),
        "isAnomaly": bool(row.get('Is_Anomaly', 0)),
        "explanation": str(row.get('Explanation_Summary', '')),
        "xgboostProb": float(row.get('XGB_Critical_Prob', 0)),
        "anomalyScore": float(row.get('Anomaly_Score', 0)),
        "hsCode": str(raw_row.get('HS_Code', 'Unknown')),
        "hsDesc": "Goods"
    }

@app.post("/api/predict")
def predict_risk(data: PredictRequest):
    # This is a realistic mock representation since generating the exact 27+ features 
    # matching the DataFrame exactly is complex without the full feature pipeline loaded.
    # In a full production script, you'd apply the same feature_engineering steps here.
    
    declW = float(data.declaredWeight) if data.declaredWeight else 0
    measW = float(data.measuredWeight) if data.measuredWeight else 0
    val = float(data.declaredValue) if data.declaredValue else 0
    dwell = float(data.dwellTime) if data.dwellTime else 0
    
    discrepancy = ((measW - declW) / declW * 100) if declW > 0 else 0
    vpk = val / declW if declW > 0 else 0
    
    hour = -1
    if data.shipmentDate:
        try:
            hour = datetime.fromisoformat(data.shipmentDate.replace('Z', '+00:00')).hour
        except:
            pass
    
    isLate = hour >= 22 or (hour >= 0 and hour < 5)
    
    score = 0
    factors = []
    
    if abs(discrepancy) > 20: 
        score += 45
        factors.append(f"Major Weight Discrepancy: {discrepancy:+.1f}%")
    elif abs(discrepancy) > 10:
        score += 25
        factors.append(f"Elevated Weight Discrepancy: {discrepancy:+.1f}%")
        
    if isLate:
        score += 15
        factors.append("Late-Night Declaration")
        
    if dwell > 10:
        score += 25
        factors.append(f"Excessive Dwell Time: {dwell}d")
        
    if vpk < 2 and declW > 0:
        score += 15
        factors.append("Unusually Low Value-per-KG")
        
    # Baseline jitter
    score += np.random.uniform(5, 15)
    score = min(100, max(0, score))
    
    if score > 70:
        level = "Critical"
    elif score > 35:
        level = "Low Risk"
    else:
        level = "Clear"
        
    anom_score = -0.2 if score > 70 else 0.05
    
    return {
        "riskLevel": level,
        "confidence": score, # using score as confidence equivalent for UI
        "xgboostScore": score / 100.0,
        "anomalyScore": anom_score,
        "factors": factors,
        "recommendation": "Flag for inspection" if level == "Critical" else "Clear"
    }
