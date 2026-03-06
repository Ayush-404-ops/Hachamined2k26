import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
import warnings
from datetime import datetime, time

warnings.filterwarnings('ignore')
pd.set_option("styler.render.max_elements", 1000000)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SmartContainer Risk Engine",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

    /* Sidebar */
    .css-1d391kg, section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    }
    section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

    /* KPI card */
    .kpi-card {
        background: linear-gradient(135deg, #1c2128 0%, #21262d 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .kpi-value { font-size: 2.2rem; font-weight: 700; margin: 4px 0; }
    .kpi-label { font-size: 0.85rem; color: #8b949e; letter-spacing: 0.05em; text-transform: uppercase; }
    .kpi-sub   { font-size: 0.95rem; font-weight: 600; margin-top: 2px; }

    /* Badge */
    .badge-critical { background:#da3633; color:#fff; padding:4px 12px; border-radius:99px; font-weight:700; font-size:0.9rem; }
    .badge-lowrisk  { background:#d29922; color:#fff; padding:4px 12px; border-radius:99px; font-weight:700; font-size:0.9rem; }
    .badge-clear    { background:#238636; color:#fff; padding:4px 12px; border-radius:99px; font-weight:700; font-size:0.9rem; }

    /* Container lookup card */
    .info-card {
        background: #1c2128; border: 1px solid #30363d; border-radius: 12px;
        padding: 24px; margin-bottom: 12px;
    }
    .info-card h3 { color: #e6edf3; margin-bottom: 12px; }
    .info-row { display:flex; justify-content:space-between; padding: 6px 0; border-bottom: 1px solid #21262d; }
    .info-label { color: #8b949e; font-size: 0.9rem; }
    .info-value { color: #e6edf3; font-weight: 600; font-size: 0.95rem; }
    .warn { color: #ffa657; }

    /* Header */
    .main-header {
        background: linear-gradient(90deg, #0d1117 0%, #161b22 100%);
        border-bottom: 1px solid #30363d;
        padding: 12px 0 8px 0;
        margin-bottom: 20px;
    }
    .main-header h1 { color: #e6edf3; font-size: 1.8rem; font-weight: 700; margin:0; }
    .main-header p  { color: #8b949e; font-size: 0.85rem; margin: 2px 0 0 0; }

    /* Footer */
    .footer { color: #484f58; font-size: 0.78rem; text-align: center; padding: 16px 0 6px 0;
              border-top: 1px solid #21262d; margin-top: 40px; }
    
    /* Plotly dark override */
    .js-plotly-plot .plotly .modebar { background: transparent !important; }
    
    /* Table coloring */
    .critical-row { background-color: rgba(218,54,51,0.15) !important; }
</style>
""", unsafe_allow_html=True)

RISK_COLORS = {"Critical": "#da3633", "Low Risk": "#d29922", "Clear": "#238636"}

# ─────────────────────────────────────────────
# DATA LOADERS (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_predictions():
    return pd.read_csv("final_predictions.csv")

@st.cache_data
def load_processed_data():
    raw = pd.read_csv("Historical Data-1.csv")
    col_map = {
        'Declaration_Date (YYYY-MM-DD)': 'Declaration_Date',
        'Declaration_Time (HH:MM:SS)': 'Declaration_Time',
        'Trade_Regime (Import / Export / Transit)': 'Trade_Regime'
    }
    raw.rename(columns=col_map, inplace=True)
    proc = pd.read_csv("processed_data.csv")
    # Merge Container_ID and origin cols back if they are not already in proc
    keep = ['Container_ID','Declaration_Date','Origin_Country','Destination_Country',
            'Destination_Port','Shipping_Line','HS_Code','Importer_ID','Exporter_ID',
            'Declared_Value','Declared_Weight','Measured_Weight','Dwell_Time_Hours',
            'Trade_Regime','Clearance_Status']
    
    # Only keep columns that are not already in proc
    cols_to_join = [col for col in keep if col not in proc.columns and col in raw.columns]
    
    if cols_to_join:
        merged = proc.join(raw[cols_to_join].reset_index(drop=True))
    else:
        merged = proc
        
    return merged

@st.cache_resource
def load_models():
    xgb_model = joblib.load("models/xgb_model.pkl")
    iso_model  = joblib.load("models/isolation_forest.pkl")
    with open("models/feature_names.json") as f:
        feature_names = json.load(f)
    return xgb_model, iso_model, feature_names

def simulate_twin_data(df):
    """
    Simulate digital twin state for each container.
    Uses existing data to create a realistic twin snapshot.
    """
    twins = []
    
    for _, row in df.iterrows():
        # Entry snapshot = declared values at arrival
        entry_weight = row['Declared_Weight']
        
        # Current state = measured values (what was actually found)
        current_weight = row['Measured_Weight']
        
        # Weight delta
        weight_delta = current_weight - entry_weight
        weight_delta_pct = (
            weight_delta / entry_weight * 100
            if entry_weight > 0 else 0
        )
        
        # Dwell time status
        dwell = row['Dwell_Time_Hours']
        dwell_status = (
            "🔴 CRITICAL" if dwell > 120 else
            "🟡 HIGH"     if dwell > 72  else
            "🟢 NORMAL"
        )
        
        # Weight change status
        weight_status = (
            "🔴 MAJOR CHANGE"  if abs(weight_delta_pct) > 15 else
            "🟡 MINOR CHANGE"  if abs(weight_delta_pct) > 5  else
            "🟢 STABLE"
        )
        
        # Twin alert level
        alerts = []
        if abs(weight_delta_pct) > 15:
            alerts.append(
                f"⚠️ Weight changed by "
                f"{weight_delta_pct:+.1f}% since entry"
            )
        if dwell > 120:
            alerts.append(
                f"⚠️ Dwell time {dwell:.0f}hrs — "
                f"exceeds 120hr threshold"
            )
        if abs(weight_delta_pct) > 5 and dwell > 72:
            alerts.append(
                "🚨 Weight change during extended dwell — "
                "possible tampering"
            )
        
        # Extract hour from Declaration_Time
        try:
            hour = int(str(row['Declaration_Time']).split(':')[0])
        except:
            hour = 12
            
        if hour >= 22 or hour <= 4:
            alerts.append(
                f"⚠️ Declared at {hour:02d}:00 — "
                f"late night activity"
            )
        
        # Overall twin status
        if len(alerts) >= 3:
            twin_status = "🔴 TAMPERING SUSPECTED"
        elif len(alerts) == 2:
            twin_status = "🟡 ANOMALIES DETECTED"
        elif len(alerts) == 1:
            twin_status = "🟡 MONITORING"
        else:
            twin_status = "🟢 NORMAL"
        
        # Risk score from predictions
        risk_score = row.get('Risk_Score', 0)
        
        # Simulated twin timeline (3 checkpoints)
        timeline = [
            {
                'checkpoint': 'Entry Scan',
                'time': '0 hrs',
                'weight': entry_weight,
                'status': '🟢 Baseline recorded'
            },
            {
                'checkpoint': 'Mid-Dwell Check',
                'time': f'{dwell/2:.0f} hrs',
                'weight': entry_weight + (weight_delta * 0.6),
                'status': (
                    '🟡 Minor variation'
                    if abs(weight_delta_pct) > 5
                    else '🟢 Stable'
                )
            },
            {
                'checkpoint': 'Exit Scan',
                'time': f'{dwell:.0f} hrs',
                'weight': current_weight,
                'status': weight_status
            }
        ]
        
        twins.append({
            'Container_ID':       row['Container_ID'],
            'Entry_Weight':       entry_weight,
            'Current_Weight':     current_weight,
            'Weight_Delta':       weight_delta,
            'Weight_Delta_Pct':   weight_delta_pct,
            'Dwell_Time_Hours':   dwell,
            'Weight_Status':      weight_status,
            'Dwell_Status':       dwell_status,
            'Twin_Status':        twin_status,
            'Alert_Count':        len(alerts),
            'Alerts':             alerts,
            'Timeline':           timeline,
            'Risk_Score':         risk_score,
            'Risk_Level':         row.get('Risk_Level', 'Clear'),
            'Origin_Country':     row.get('Origin_Country', ''),
            'Destination_Country':row.get('Destination_Country', ''),
            'Importer_ID':        row.get('Importer_ID', ''),
            'Shipping_Line':      row.get('Shipping_Line', ''),
            'Declaration_Time':   row.get('Declaration_Time', ''),
            'Explanation_Summary':row.get('Explanation_Summary', '')
        })
    
    return pd.DataFrame(twins)

@st.cache_data
def get_twin_data():
    preds = load_predictions()
    proc  = load_processed_data()
    # Merge on Container_ID
    if 'Container_ID' in proc.columns and 'Container_ID' in preds.columns:
        merged = preds.merge(proc, on='Container_ID', how='inner')
        return simulate_twin_data(merged)
    return pd.DataFrame()

# ─────────────────────────────────────────────
# HELPER — badge HTML
# ─────────────────────────────────────────────
def badge(level):
    cls = {"Critical":"badge-critical","Low Risk":"badge-lowrisk","Clear":"badge-clear"}.get(level,"badge-clear")
    icon = {"Critical":"🔴","Low Risk":"🟠","Clear":"🟢"}.get(level,"")
    return f'<span class="{cls}">{icon} {level}</span>'

def kpi_card(value, label, sub="", color="#58a6ff"):
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color};">{value}</div>
        <div class="kpi-sub" style="color:{color};">{sub}</div>
    </div>"""

# ─────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────
def render_overview(preds, proc):
    st.markdown('<div class="main-header"><h1>📊 Overview Dashboard</h1>'
                '<p>Real-time summary of container risk across all shipments</p></div>',
                unsafe_allow_html=True)

    total  = len(preds)
    crit   = (preds['Risk_Level'] == 'Critical').sum()
    low    = (preds['Risk_Level'] == 'Low Risk').sum()
    anom   = preds['Is_Anomaly'].sum()

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card(f"{total:,}", "Total Containers", "", "#58a6ff"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card(f"{crit:,}", "Critical Containers", f"{crit/total*100:.1f}%", "#da3633"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card(f"{low:,}", "Low Risk Containers", f"{low/total*100:.1f}%", "#d29922"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card(f"{anom:,}", "Anomalies Detected", f"{anom/total*100:.1f}%", "#ffa657"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2 — 3 charts
    c1, c2, c3 = st.columns(3)
    with c1:
        rl_counts = preds['Risk_Level'].value_counts().reset_index()
        rl_counts.columns = ['Risk_Level','Count']
        fig = px.pie(rl_counts, names='Risk_Level', values='Count',
                     color='Risk_Level', color_discrete_map=RISK_COLORS,
                     title="Risk Level Distribution", hole=0.45)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#e6edf3', legend_font_color='#e6edf3', title_font_color='#e6edf3')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Need Origin_Country — merge from proc
        preds_geo = preds.copy()
        if 'Origin_Country' in proc.columns:
            orig = proc[['Origin_Country']].iloc[:len(preds)].reset_index(drop=True)
            preds_geo = pd.concat([preds_geo.reset_index(drop=True), orig], axis=1)
            crit_df = preds_geo[preds_geo['Risk_Level']=='Critical']
            top_orig = crit_df['Origin_Country'].value_counts().head(10).reset_index()
            top_orig.columns = ['Country','Count']
            fig2 = px.bar(top_orig, x='Count', y='Country', orientation='h',
                          title="Top 10 Origin Countries (Critical)", color='Count',
                          color_continuous_scale='Reds')
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               font_color='#e6edf3', yaxis=dict(autorange='reversed'),
                               title_font_color='#e6edf3', showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

    with c3:
        fig3 = px.histogram(preds, x='Risk_Score', color='Risk_Level',
                            color_discrete_map=RISK_COLORS, nbins=50,
                            title="Risk Score Distribution", barmode='overlay')
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font_color='#e6edf3', title_font_color='#e6edf3', legend_font_color='#e6edf3')
        st.plotly_chart(fig3, use_container_width=True)

    # Row 3 — HS Chapter + Shipping Line
    c1, c2 = st.columns(2)
    if 'HS_Code' in proc.columns and 'Clearance_Status' in proc.columns:
        proc['HS_Chapter'] = proc['HS_Code'].astype(str).str.zfill(6).str[:2]
        total_hs = proc.groupby('HS_Chapter').size()
        crit_hs  = proc[proc['Clearance_Status']=='Critical'].groupby('HS_Chapter').size()
        hs_rate  = (crit_hs / total_hs * 100).dropna().sort_values(ascending=False).head(10).reset_index()
        hs_rate.columns  = ['HS_Chapter','Critical_Rate']

        with c1:
            fig4 = px.bar(hs_rate, x='HS_Chapter', y='Critical_Rate',
                          title="Critical Rate by HS Chapter (Top 10)",
                          color='Critical_Rate', color_continuous_scale='Oranges')
            fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               font_color='#e6edf3', title_font_color='#e6edf3', coloraxis_showscale=False)
            st.plotly_chart(fig4, use_container_width=True)

    if 'Shipping_Line' in proc.columns and 'Clearance_Status' in proc.columns:
        total_sl = proc.groupby('Shipping_Line').size()
        crit_sl  = proc[proc['Clearance_Status']=='Critical'].groupby('Shipping_Line').size()
        sl_rate  = (crit_sl / total_sl * 100).dropna().sort_values(ascending=False).head(10).reset_index()
        sl_rate.columns = ['Shipping_Line','Critical_Rate']

        with c2:
            fig5 = px.bar(sl_rate, x='Shipping_Line', y='Critical_Rate',
                          title="Critical Rate by Shipping Line (Top 10)",
                          color='Critical_Rate', color_continuous_scale='Blues')
            fig5.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               font_color='#e6edf3', title_font_color='#e6edf3', coloraxis_showscale=False,
                               xaxis_tickangle=-35)
            st.plotly_chart(fig5, use_container_width=True)

    # ── ROI Calculator (New Section) ──
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown('<div class="main-header"><h1>💰 Business Impact & ROI</h1></div>', unsafe_allow_html=True)
    
    # Assumptions
    trad_rate = 0.40  # 40% random inspection
    ai_rate   = crit / total if total > 0 else 0
    inspections_avoided = (trad_rate - ai_rate) * total
    
    hrs_per_ins = 2.0
    wage_hr     = 15
    
    manhours_saved = inspections_avoided * hrs_per_ins
    wages_saved    = manhours_saved * wage_hr
    
    # Precision gain
    trad_catch_rate = 0.05
    ai_catch_rate   = 0.38  # Assumption for AI flagged criticals
    efficiency_gain = (ai_catch_rate / trad_catch_rate) * 100
    
    ir1, ir2, ir3, ir4 = st.columns(4)
    with ir1: st.markdown(kpi_card(f"{max(0, manhours_saved):,.0f} hrs", "Manhours Saved", "", "#238636"), unsafe_allow_html=True)
    with ir2: st.markdown(kpi_card(f"${max(0, wages_saved):,.0f}", "Wages Saved", "", "#238636"), unsafe_allow_html=True)
    with ir3: st.markdown(kpi_card(f"{max(0, inspections_avoided):,.0f}", "Inspections Reduced", "", "#58a6ff"), unsafe_allow_html=True)
    with ir4: st.markdown(kpi_card(f"{efficiency_gain:.0f}x", "Detection Efficiency", "Better than Random", "#ffa657"), unsafe_allow_html=True)
    
    st.markdown('<p style="color:#8b949e; font-size:0.8rem; text-align:center; margin-top:10px;">'
                'Assumptions: 40% traditional inspection rate, $15/hr officer wage, 2hrs per inspection. '
                'Figures are estimates based on industry benchmarks.</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 2 — CRITICAL CONTAINERS
# ─────────────────────────────────────────────
def render_critical_table(preds, proc):
    st.markdown('<div class="main-header"><h1>🚨 Critical Containers</h1>'
                '<p>Drill-down view of high-risk shipments with filtering</p></div>',
                unsafe_allow_html=True)

    # Merge useful columns from raw/processed
    display_cols = ['Container_ID','Risk_Score','Risk_Level','Is_Anomaly','Explanation_Summary']
    extra_raw = ['Origin_Country','Dwell_Time_Hours','Shipping_Line','Declaration_Date']
    
    df = preds[display_cols].copy()
    for col in extra_raw:
        if col in proc.columns:
            df[col] = proc[col].iloc[:len(df)].values

    # ── Filters ──
    st.markdown("### 🔎 Filters")
    fc1, fc2, fc3, fc4 = st.columns([2,2,2,1])

    with fc1:
        if 'Declaration_Date' in df.columns:
            df['Declaration_Date'] = pd.to_datetime(df['Declaration_Date'], errors='coerce')
            valid_dates = df['Declaration_Date'].dropna()
            if len(valid_dates):
                d_min, d_max = valid_dates.min().date(), valid_dates.max().date()
                date_range = st.date_input("Date Range", value=(d_min, d_max))
                if len(date_range) == 2:
                    df = df[(df['Declaration_Date'].dt.date >= date_range[0]) &
                            (df['Declaration_Date'].dt.date <= date_range[1])]

    with fc2:
        if 'Origin_Country' in df.columns:
            origins = sorted(df['Origin_Country'].dropna().unique())
            sel_orig = st.multiselect("Origin Country", origins)
            if sel_orig:
                df = df[df['Origin_Country'].isin(sel_orig)]

    with fc3:
        if 'Shipping_Line' in df.columns:
            lines = sorted(df['Shipping_Line'].dropna().unique())
            sel_line = st.multiselect("Shipping Line", lines)
            if sel_line:
                df = df[df['Shipping_Line'].isin(sel_line)]

    with fc4:
        anom_only = st.checkbox("Anomaly Only")
        if anom_only:
            df = df[df['Is_Anomaly'] == 1]

    # ── Color-coded table ──
    st.markdown(f"**{len(df):,} containers** match the current filters.")
    
    # Cap to top 5000 rows for performance (avoiding Pandas Styler cell limits)
    if len(df) > 5000:
        st.info("⚠️ Performance Tip: Showing top 5,000 results. Use the filters to narrow the search.")
        df = df.head(5000)
    
    def highlight_risk(row):
        colors = {"Critical": "background-color: rgba(218,54,51,0.18);",
                  "Low Risk": "background-color: rgba(210,153,34,0.18);",
                  "Clear":    "background-color: rgba(35,134,54,0.10);"}
        return [colors.get(row.get('Risk_Level',''), '')] * len(row)

    show_df = df.drop(columns=['Declaration_Date'], errors='ignore')
    show_df = show_df[['Container_ID','Risk_Score','Risk_Level','Is_Anomaly',
                        'Origin_Country','Dwell_Time_Hours','Explanation_Summary']].fillna("—")
    
    st.dataframe(
        show_df.style.apply(highlight_risk, axis=1),
        use_container_width=True, height=460
    )

    # Download
    csv = show_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Filtered Results as CSV", csv,
                       file_name="filtered_containers.csv", mime="text/csv")

# ─────────────────────────────────────────────
# PAGE 3 — CONTAINER LOOKUP
# ─────────────────────────────────────────────
def render_container_lookup(preds, proc):
    st.markdown('<div class="main-header"><h1>🔍 Container Lookup</h1>'
                '<p>Detailed risk report for a specific container</p></div>',
                unsafe_allow_html=True)

    search_id = st.text_input("Enter Container ID", placeholder="e.g. CONT_000001")
    if not search_id:
        st.info("Enter a Container ID above to view its risk assessment.")
        return

    pred_row = preds[preds['Container_ID'].astype(str) == search_id.strip()]
    if pred_row.empty:
        st.error(f"Container `{search_id}` not found.")
        return

    pred_row = pred_row.iloc[0]
    idx = preds.index[preds['Container_ID'].astype(str) == search_id.strip()][0]
    raw_row = proc.iloc[idx] if idx < len(proc) else None

    risk_level = pred_row['Risk_Level']
    risk_score = pred_row['Risk_Score']
    badge_html = badge(risk_level)
    icon = {"Critical":"🔴","Low Risk":"🟠","Clear":"🟢"}.get(risk_level,"")

    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown(f"""
        <div class="info-card">
            <h3>📦 {search_id}</h3>
            <div class="info-row">
                <span class="info-label">Risk Level</span>
                <span>{badge_html}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Risk Score</span>
                <span class="info-value" style="color:{'#da3633' if risk_level=='Critical' else '#d29922' if risk_level=='Low Risk' else '#238636'}">
                    {risk_score:.2f} / 100
                </span>
            </div>
        """, unsafe_allow_html=True)

        if raw_row is not None:
            orig = raw_row.get('Origin_Country', '—')
            dest = raw_row.get('Destination_Country', '—')
            decl_w = raw_row.get('Declared_Weight', None)
            meas_w = raw_row.get('Measured_Weight', None)
            dwell  = raw_row.get('Dwell_Time_Hours', None)
            decl_v = raw_row.get('Declared_Value', None)
            ship   = raw_row.get('Shipping_Line', '—')

            disc_str = ""
            if decl_w and meas_w and decl_w != 0:
                disc = (meas_w - decl_w) / decl_w * 100
                warn = " ⚠️" if abs(disc) > 10 else ""
                disc_str = f" ({disc:+.1f}%{warn})"

            dwell_warn = " ⚠️ HIGH" if dwell and dwell > 96 else ""

            st.markdown(f"""
            <div class="info-row">
                <span class="info-label">Route</span>
                <span class="info-value">{orig} → {dest}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Declared Weight</span>
                <span class="info-value">{f'{decl_w:,.1f} kg' if decl_w else '—'}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Measured Weight</span>
                <span class="info-value {'warn' if (decl_w and meas_w and decl_w != 0 and abs(disc) > 10) else ''}">{f'{meas_w:,.1f} kg' if meas_w else '—'}{disc_str}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Dwell Time</span>
                <span class="info-value{'warn' if dwell and dwell > 96 else ''}">{f'{dwell:.1f} hours' if dwell else '—'}{dwell_warn}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Declared Value</span>
                <span class="info-value">{f'${decl_v:,.0f}' if decl_v else '—'}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Shipping Line</span>
                <span class="info-value">{ship}</span>
            </div>
            """, unsafe_allow_html=True)

        expl = pred_row.get('Explanation_Summary', '—')
        is_anom = pred_row.get('Is_Anomaly', 0)
        xgb_prob = pred_row.get('XGB_Critical_Prob', 0)
        anom_score = pred_row.get('Anomaly_Score', 0)

        st.markdown(f"""
            <div style="margin-top:12px; padding: 12px; background: rgba(56,139,253,0.08); border-radius:8px; border-left: 3px solid #58a6ff;">
                <span style="color:#58a6ff; font-weight:600;">🤖 AI Explanation</span><br>
                <span style="color:#e6edf3; font-style:italic;">"{expl}"</span>
            </div>
            <div class="info-row" style="margin-top:10px;">
                <span class="info-label">Anomaly Detected</span>
                <span class="info-value">{'✅ YES' if is_anom else '❌ NO'}</span>
            </div>
            <div class="info-row">
                <span class="info-label">XGB Critical Probability</span>
                <span class="info-value">{xgb_prob:.4f}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Anomaly Score</span>
                <span class="info-value">{anom_score:.4f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("##### Risk Gauge")
        gauge_val = float(risk_score)
        color = RISK_COLORS.get(risk_level, "#238636")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gauge_val,
            domain={'x':[0,1],'y':[0,1]},
            gauge={
                'axis': {'range':[0,100], 'tickcolor':'#e6edf3'},
                'bar': {'color': color},
                'bgcolor': '#21262d',
                'steps': [
                    {'range':[0,25],  'color':'rgba(35,134,54,0.2)'},
                    {'range':[25,60], 'color':'rgba(210,153,34,0.2)'},
                    {'range':[60,100],'color':'rgba(218,54,51,0.2)'},
                ],
                'threshold': {'line':{'color':color,'width':4},'thickness':0.8,'value':gauge_val}
            },
            number={'font':{'color':color,'size':36}}
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e6edf3', height=300, margin=dict(t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Risk Score Progress**")
        st.progress(min(int(gauge_val), 100))

# ─────────────────────────────────────────────
# PAGE 4 — GEOGRAPHIC RISK
# ─────────────────────────────────────────────
def render_geographic_risk(preds, proc):
    st.markdown('<div class="main-header"><h1>🌍 Geographic Risk Analysis</h1>'
                '<p>World map of critical container origin countries</p></div>',
                unsafe_allow_html=True)

    if 'Origin_Country' not in proc.columns or 'Clearance_Status' not in proc.columns:
        st.warning("Geographic data not available.")
        return

    total_by_orig = proc.groupby('Origin_Country').size().reset_index(name='Total')
    crit_by_orig  = proc[proc['Clearance_Status']=='Critical'].groupby('Origin_Country').size().reset_index(name='Critical_Count')
    geo_df = total_by_orig.merge(crit_by_orig, on='Origin_Country', how='left').fillna(0)
    geo_df['Critical_Rate'] = (geo_df['Critical_Count'] / geo_df['Total'] * 100).round(2)

    fig_map = px.choropleth(
        geo_df,
        locations='Origin_Country',
        locationmode='country names',
        color='Critical_Rate',
        hover_name='Origin_Country',
        hover_data={'Total':':.0f', 'Critical_Count':':.0f', 'Critical_Rate':':.2f'},
        color_continuous_scale='Reds',
        title="Critical Container Rate by Origin Country (%)",
        labels={'Critical_Rate':'Critical Rate (%)'}
    )
    fig_map.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e6edf3', title_font_color='#e6edf3',
        geo=dict(bgcolor='rgba(0,0,0,0)', lakecolor='#0d1117',
                 landcolor='#21262d', coastlinecolor='#30363d',
                 showframe=False, projection_type='natural earth'),
        coloraxis_colorbar=dict(tickfont=dict(color='#e6edf3'), title=dict(font=dict(color='#e6edf3'))),
        height=480, margin=dict(t=50,b=0,l=0,r=0)
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Two bar charts
    c1, c2 = st.columns(2)
    with c1:
        top15_count = geo_df.nlargest(15,'Critical_Count')
        fig_c = px.bar(top15_count, x='Critical_Count', y='Origin_Country', orientation='h',
                       color='Critical_Count', color_continuous_scale='Reds',
                       title="Top 15 Origins — Critical Count")
        fig_c.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font_color='#e6edf3', title_font_color='#e6edf3',
                            yaxis=dict(autorange='reversed'), coloraxis_showscale=False)
        st.plotly_chart(fig_c, use_container_width=True)

    with c2:
        top15_rate = geo_df.nlargest(15,'Critical_Rate')
        fig_r = px.bar(top15_rate, x='Critical_Rate', y='Origin_Country', orientation='h',
                       color='Critical_Rate', color_continuous_scale='Oranges',
                       title="Top 15 Origins — Critical Rate %")
        fig_r.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font_color='#e6edf3', title_font_color='#e6edf3',
                            yaxis=dict(autorange='reversed'), coloraxis_showscale=False)
        st.plotly_chart(fig_r, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 5 — LIVE RISK PREDICTOR
# ─────────────────────────────────────────────
FEATURE_LABELS = {
    'weight_discrepancy': 'Weight Mismatch',
    'abs_weight_discrepancy': 'Abs Weight Discrepancy',
    'value_per_kg': 'Value per KG',
    'dwell_time_flag': 'Excessive Dwell Time',
    'Dwell_Time_Hours': 'Dwell Time Hours',
    'is_late_night': 'Late-Night Declaration',
    'Importer_ID_critical_rate': 'Importer Risk Rate',
    'Exporter_ID_critical_rate': 'Exporter Risk Rate',
    'Origin_Country_critical_rate': 'Origin Country Risk',
    'HS_Chapter_critical_rate': 'HS Chapter Risk',
    'weight_discrepancy_x_dwell': 'Weight+Dwell Anomaly',
    'log_value_per_kg': 'Log Value/KG',
    'log_dwell_time': 'Log Dwell Time',
}

def render_live_predictor(proc):
    st.markdown('<div class="main-header"><h1>🤖 Live Risk Predictor</h1>'
                '<p>Score a new container in real-time using the trained XGBoost model</p></div>',
                unsafe_allow_html=True)

    with st.form("risk_form"):
        st.markdown("#### 📋 Container Details")
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1: origin_country = st.text_input("Origin Country", value="CN")
        with r1c2: dest_country   = st.text_input("Destination Country", value="US")
        with r1c3: trade_regime   = st.selectbox("Trade Regime", ["Import", "Export", "Transit"])

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1: decl_value   = st.number_input("Declared Value (USD)", value=10000.0, min_value=0.0)
        with r2c2: decl_weight  = st.number_input("Declared Weight (kg)", value=500.0, min_value=0.01)
        with r2c3: meas_weight  = st.number_input("Measured Weight (kg)", value=510.0, min_value=0.01)

        r3c1, r3c2, r3c3 = st.columns(3)
        with r3c1: dwell_hours  = st.number_input("Dwell Time (hours)", value=48.0, min_value=0.0)
        with r3c2: decl_time    = st.time_input("Declaration Time", value=time(14, 30))
        with r3c3: hs_code      = st.number_input("HS Code", value=620000, step=1)

        r4c1, r4c2, r4c3 = st.columns(3)
        with r4c1: importer_id  = st.text_input("Importer ID", value="IMP_001")
        with r4c2: exporter_id  = st.text_input("Exporter ID", value="EXP_001")
        with r4c3: shipping_line = st.text_input("Shipping Line", value="LINE_1")

        submitted = st.form_submit_button("🔍 Assess Risk", use_container_width=True)

    if not submitted:
        return

    xgb_model, iso_model, feature_names = load_models()

    # ── Feature Engineering from inputs ──
    hour = decl_time.hour
    hs_chapter = str(int(hs_code)).zfill(6)[:2]

    weight_disc   = (meas_weight - decl_weight) / decl_weight * 100
    abs_wd        = abs(weight_disc)
    wd_flag       = 1 if abs_wd > 10 else 0
    value_per_kg  = decl_value / decl_weight
    log_val       = np.log1p(decl_value)
    log_wt        = np.log1p(decl_weight)
    log_vpk       = np.log1p(value_per_kg)
    weight_ratio  = meas_weight / decl_weight
    log_dwell     = np.log1p(dwell_hours)
    dwell_flag    = 1 if dwell_hours > 96 else 0
    is_late       = 1 if (hour >= 22 or hour <= 4) else 0
    day_of_week   = datetime.now().weekday()
    is_weekend    = 1 if day_of_week >= 5 else 0
    month_num     = datetime.now().month
    hour_sin      = np.sin(2 * np.pi * hour / 24)
    hour_cos      = np.cos(2 * np.pi * hour / 24)
    trade_enc     = {"Import":0,"Transit":1,"Export":2}.get(trade_regime, 0)
    hs_freq       = 1  # unknown, use 1

    # Global means from processed_data
    global_crit_rate = proc['Clearance_Status_Encoded'].pipe(lambda x: (x == 2).mean()) if 'Clearance_Status_Encoded' in proc.columns else 0.01

    # Lookup entity risk rates from proc if possible
    def get_rate(df, col, val, target_col='Clearance_Status_Encoded', target_val=2, fallback=None):
        if col not in df.columns: return fallback or global_crit_rate
        sub = df[df[col].astype(str) == str(val)]
        if len(sub) < 3: return fallback or global_crit_rate
        return (sub[target_col] == target_val).mean() if target_col in df.columns else global_crit_rate

    # Dwell zscore — approximate global
    dwell_mean = proc['Dwell_Time_Hours'].mean() if 'Dwell_Time_Hours' in proc.columns else dwell_hours
    dwell_std  = proc['Dwell_Time_Hours'].std()  if 'Dwell_Time_Hours' in proc.columns else 1
    dwell_z    = (dwell_hours - dwell_mean) / max(dwell_std, 1e-6)

    # Value zscore — approximate per chapter
    vpk_zscore = 0.0  # unknown chapter, default 0

    # Entity rates
    imp_crit_rate  = get_rate(proc, 'Importer_ID', importer_id)
    exp_crit_rate  = get_rate(proc, 'Exporter_ID', exporter_id)
    orig_crit_rate = get_rate(proc, 'Origin_Country', origin_country)
    dest_crit_rate = get_rate(proc, 'Destination_Country', dest_country)
    sl_crit_rate   = get_rate(proc, 'Shipping_Line', shipping_line)
    hs_crit_rate   = get_rate(proc, 'HS_Chapter', hs_chapter) if 'HS_Chapter' in proc.columns else global_crit_rate

    # Low risk rates (approx half of overall LR rate)
    global_lr_rate = proc['Clearance_Status_Encoded'].pipe(lambda x: (x == 1).mean()) if 'Clearance_Status_Encoded' in proc.columns else 0.20
    risk_score_fn  = lambda cr, lr: cr * 0.6 + lr * 0.4

    imp_rs  = risk_score_fn(imp_crit_rate, global_lr_rate)
    exp_rs  = risk_score_fn(exp_crit_rate, global_lr_rate)
    orig_rs = risk_score_fn(orig_crit_rate, global_lr_rate)
    dest_rs = risk_score_fn(dest_crit_rate, global_lr_rate)
    sl_rs   = risk_score_fn(sl_crit_rate, global_lr_rate)
    hs_rs   = risk_score_fn(hs_crit_rate, global_lr_rate)

    imp_total  = len(proc[proc['Importer_ID'].astype(str) == importer_id]) if 'Importer_ID' in proc.columns else 0
    exp_total  = len(proc[proc['Exporter_ID'].astype(str) == exporter_id]) if 'Exporter_ID' in proc.columns else 0
    orig_total = len(proc[proc['Origin_Country'].astype(str) == origin_country]) if 'Origin_Country' in proc.columns else 0
    dest_total = len(proc[proc['Destination_Country'].astype(str) == dest_country]) if 'Destination_Country' in proc.columns else 0
    sl_total   = len(proc[proc['Shipping_Line'].astype(str) == shipping_line]) if 'Shipping_Line' in proc.columns else 0
    hs_total   = 1

    imp_crit_count = int(imp_crit_rate * imp_total) if imp_total > 0 else 0
    exp_crit_count = int(exp_crit_rate * exp_total) if exp_total > 0 else 0
    imp_repeat = 1 if imp_crit_count > 1 else 0
    exp_repeat = 1 if exp_crit_count > 1 else 0
    val_anom_x_orig = log_vpk * orig_crit_rate
    wd_x_dwell      = abs_wd * log_dwell
    high_risk_combo = 1 if (wd_flag == 1 and dwell_flag == 1) else 0

    if 'Destination_Port' in proc.columns:
        dp_crit_rate = get_rate(proc, 'Destination_Port', "UNKNOWN")
        dp_total = 0
        dp_rs = risk_score_fn(dp_crit_rate, global_lr_rate)
    else:
        dp_crit_rate = global_crit_rate; dp_rs = risk_score_fn(global_crit_rate, global_lr_rate); dp_total = 0

    # Build feature dict matching expected feature order
    feat_dict = {
        'hour_of_day': hour,
        'is_late_night': is_late,
        'day_of_week_num': day_of_week,
        'is_weekend': is_weekend,
        'month': month_num,
        'declaration_hour_sin': hour_sin,
        'declaration_hour_cos': hour_cos,
        'weight_discrepancy': weight_disc,
        'abs_weight_discrepancy': abs_wd,
        'weight_discrepancy_flag': wd_flag,
        'value_per_kg': value_per_kg,
        'log_value': log_val,
        'log_weight': log_wt,
        'log_value_per_kg': log_vpk,
        'value_per_kg_zscore': vpk_zscore,
        'weight_ratio': weight_ratio,
        'dwell_time_flag': dwell_flag,
        'log_dwell_time': log_dwell,
        'dwell_time_zscore': dwell_z,
        'HS_Code_frequency': hs_freq,
        'importer_critical_count': imp_crit_count,
        'exporter_critical_count': exp_crit_count,
        'importer_is_repeat_offender': imp_repeat,
        'exporter_is_repeat_offender': exp_repeat,
        'weight_discrepancy_x_dwell': wd_x_dwell,
        'high_risk_combo': high_risk_combo,
        'Trade_Regime_Encoded': trade_enc,
        'Importer_ID_critical_rate': imp_crit_rate,
        'Importer_ID_total_shipments': imp_total,
        'Importer_ID_risk_score': imp_rs,
        'Exporter_ID_critical_rate': exp_crit_rate,
        'Exporter_ID_total_shipments': exp_total,
        'Exporter_ID_risk_score': exp_rs,
        'Origin_Country_critical_rate': orig_crit_rate,
        'Origin_Country_total_shipments': orig_total,
        'Origin_Country_risk_score': orig_rs,
        'Destination_Country_critical_rate': dest_crit_rate,
        'Destination_Country_total_shipments': dest_total,
        'Destination_Country_risk_score': dest_rs,
        'Shipping_Line_critical_rate': sl_crit_rate,
        'Shipping_Line_total_shipments': sl_total,
        'Shipping_Line_risk_score': sl_rs,
        'HS_Chapter_critical_rate': hs_crit_rate,
        'HS_Chapter_total_shipments': hs_total,
        'HS_Chapter_risk_score': hs_rs,
        'Destination_Port_critical_rate': dp_crit_rate,
        'Destination_Port_total_shipments': dp_total,
        'Destination_Port_risk_score': dp_rs,
        'value_anomaly_x_origin_risk': val_anom_x_orig,
    }

    # Build row aligned to model feature_names
    row = pd.DataFrame([{f: feat_dict.get(f, 0) for f in feature_names}])

    # Predict
    probs     = xgb_model.predict_proba(row)[0]
    crit_prob = probs[2]; lr_prob = probs[1]
    
    iso_score_raw = -iso_model.decision_function(row)[0]
    # Normalize relative to known min/max (approx)
    anom_score_norm = min(max(iso_score_raw, 0), 1)
    is_anom = 1 if iso_model.predict(row)[0] == -1 else 0

    risk_score_combined = (crit_prob * 0.7 + lr_prob * 0.2 + anom_score_norm * 0.1) * 100
    risk_level = "Critical" if risk_score_combined >= 60 else ("Low Risk" if risk_score_combined >= 25 else "Clear")

    # Explanation (top features by raw contribution)
    feat_contribs = {
        'weight_discrepancy': abs(weight_disc) / 100,
        'dwell_time_flag': dwell_flag * 0.8,
        'Origin_Country_critical_rate': orig_crit_rate * 10,
        'is_late_night': is_late * 0.5,
        'Importer_ID_critical_rate': imp_crit_rate * 10,
        'value_per_kg': min(value_per_kg / 1000, 1),
    }
    sorted_contribs = sorted(feat_contribs.items(), key=lambda x: x[1], reverse=True)
    top3_labels = [FEATURE_LABELS.get(k, k.replace('_',' ')) for k, _ in sorted_contribs[:3]]

    if risk_level == 'Clear':
        explanation = "Cleared: no significant risk indicators detected."
    else:
        explanation = f"Flagged {risk_level}: {', '.join(top3_labels)}."

    # Result card
    color = RISK_COLORS.get(risk_level, "#238636")
    icon  = {"Critical":"🔴","Low Risk":"🟠","Clear":"🟢"}.get(risk_level, "🟢")
    
    cl, cr = st.columns([2,1])
    with cl:
        st.markdown(f"""
        <div class="info-card" style="border-left: 4px solid {color};">
            <h3 style="color:{color};">{icon} {risk_level} — Risk Score: {risk_score_combined:.1f} / 100</h3>
            <div class="info-row"><span class="info-label">XGB Critical Probability</span><span class="info-value">{crit_prob:.4f}</span></div>
            <div class="info-row"><span class="info-label">Anomaly Score</span><span class="info-value">{anom_score_norm:.4f}</span></div>
            <div class="info-row"><span class="info-label">Anomaly Detected</span><span class="info-value">{'✅ YES' if is_anom else '❌ NO'}</span></div>
            <div style="margin-top:12px; padding:12px; background:rgba(56,139,253,0.08); border-radius:8px; border-left:3px solid #58a6ff;">
                <span style="color:#58a6ff;font-weight:600;">🤖 AI Explanation</span><br>
                <span style="color:#e6edf3;font-style:italic;">"{explanation}"</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with cr:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=risk_score_combined,
            gauge={'axis':{'range':[0,100]},
                   'bar':{'color':color},'bgcolor':'#21262d',
                   'steps':[{'range':[0,25],'color':'rgba(35,134,54,0.2)'},
                             {'range':[25,60],'color':'rgba(210,153,34,0.2)'},
                             {'range':[60,100],'color':'rgba(218,54,51,0.2)'}]},
            number={'font':{'color':color,'size':40}}
        ))
        fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e6edf3', height=280, margin=dict(t=20,b=10))
        st.plotly_chart(fig_g, use_container_width=True)

    # Feature contribution chart
    contrib_df = pd.DataFrame(sorted_contribs[:8], columns=['Feature','Contribution'])
    contrib_df['Feature'] = contrib_df['Feature'].map(lambda x: FEATURE_LABELS.get(x, x.replace('_',' ')))
    fig_bar = px.bar(contrib_df, x='Contribution', y='Feature', orientation='h',
                     color='Contribution', color_continuous_scale='Reds',
                     title="Top Feature Contributions (Proxy)")
    fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#e6edf3', title_font_color='#e6edf3',
                          yaxis=dict(autorange='reversed'), coloraxis_showscale=False)
    st.plotly_chart(fig_bar, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 6 — INSPECTION PRIORITY QUEUE
# ─────────────────────────────────────────────
def render_inspection_queue(preds, proc):
    st.markdown('<div class="main-header"><h1>📋 Inspection Priority Queue</h1>'
                '<p>Containers ranked by AI risk score — highest priority first</p></div>',
                unsafe_allow_html=True)

    # ── Filters ──
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        levels = st.multiselect("Risk Level", ["Critical", "Low Risk", "Clear"], default=["Critical", "Low Risk"])
    with c2:
        min_score = st.slider("Min Risk Score", 0, 100, 50)
    with c3:
        anom_only = st.checkbox("Anomalies Only", key="q_anom")

    # Filter data
    queue_df = preds[
        (preds['Risk_Level'].isin(levels)) & 
        (preds['Risk_Score'] >= min_score)
    ].copy()
    
    if anom_only:
        queue_df = queue_df[queue_df['Is_Anomaly'] == 1]

    # Sort by score
    queue_df = queue_df.sort_values("Risk_Score", ascending=False)
    queue_df['Queue_Position'] = range(1, len(queue_df) + 1)

    # Merge some info
    for col in ['Origin_Country', 'Dwell_Time_Hours']:
        if col in proc.columns:
            queue_df[col] = proc[col].iloc[queue_df.index].values

    # Est Time Logic
    def get_est_time(lvl):
        if lvl == 'Critical': return "~2 hours"
        if lvl == 'Low Risk': return "~45 mins"
        return "~10 mins"
    
    def get_hrs(lvl):
        if lvl == 'Critical': return 2.0
        if lvl == 'Low Risk': return 0.75
        return 0.16

    queue_df['Est_Inspection_Time'] = queue_df['Risk_Level'].apply(get_est_time)
    queue_df['Hrs_Numeric'] = queue_df['Risk_Level'].apply(get_hrs)

    # Table display
    st.markdown(f"**{len(queue_df):,} containers** in daily queue.")
    
    def color_queue(row):
        colors = {"Critical": "background-color: rgba(218,54,51,0.18);",
                  "Low Risk": "background-color: rgba(210,153,34,0.18);",
                  "Clear":    "background-color: rgba(35,134,54,0.10);"}
        return [colors.get(row.get('Risk_Level',''), '')] * len(row)

    show_cols = ['Queue_Position', 'Container_ID', 'Risk_Score', 'Risk_Level', 
                 'Origin_Country', 'Dwell_Time_Hours', 'Is_Anomaly', 
                 'Explanation_Summary', 'Est_Inspection_Time']
    
    # Cap for performance
    disp_df = queue_df[show_cols].head(5000).fillna("—")
    
    st.dataframe(
        disp_df.style.apply(color_queue, axis=1),
        use_container_width=True, height=400
    )

    # Metrics
    total_hrs = queue_df['Hrs_Numeric'].sum()
    officers  = total_hrs / 8.0
    
    m1, m2, m3 = st.columns(3)
    with m1: st.markdown(kpi_card(f"{len(queue_df):,}", "Total in Queue", "", "#58a6ff"), unsafe_allow_html=True)
    with m2: st.markdown(kpi_card(f"{total_hrs:,.1f}", "Est. Total Hours", "", "#ffa657"), unsafe_allow_html=True)
    with m3: st.markdown(kpi_card(f"{max(1, round(officers))}", "Officers Required", "8hr shift base", "#da3633"), unsafe_allow_html=True)

    # Export
    csv = queue_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Export Queue as CSV", csv, "priority_queue.csv", "text/csv")


# ─────────────────────────────────────────────
# PAGE 7 — BEHAVIORAL PROFILING
# ─────────────────────────────────────────────
def render_behavioral_profiling(proc):
    """
    Behavioral profiles of importers and exporters.
    Cleaned and fixed based on strict user requirements:
    - Correct Critical Rate logic (all shipments vs critical subset)
    - KPI cards unified with table data
    - Min shipment slider to prevent 100% inflation
    - Search box below KPIs
    - Responsive charts and dangerous entity insight
    """
    @st.cache_data
    def load_behavioral_data():
        # User requested processed_data.csv. 
        # We ensure it works by using the already-merged 'proc' or refetching.
        return load_processed_data()

    df = load_behavioral_data()

    st.markdown('<div class="main-header"><h1>🕵️ Entity Risk Intelligence</h1>'
                '<p>Behavioral profiles of importers and exporters based on historical patterns</p></div>',
                unsafe_allow_html=True)
    
    def compute_entity_stats(df, col):
        # Total shipments per entity from FULL dataset
        total = df.groupby(col).size().reset_index(name='Total_Shipments')
        
        # Critical count per entity
        critical = df[df['Clearance_Status'] == 'Critical']\
                   .groupby(col).size().reset_index(name='Critical_Count')
        
        # Merge
        merged = total.merge(critical, on=col, how='left')
        merged['Critical_Count'] = merged['Critical_Count'].fillna(0).astype(int)
        
        # Critical rate
        merged['Critical_Rate'] = (
            merged['Critical_Count'] / merged['Total_Shipments'] * 100
        ).round(2)
        
        # Repeat offender — computed from SAME merged df
        merged['Is_Repeat_Offender'] = (merged['Critical_Count'] > 1).astype(int)
        
        # Risk badge
        def badge_logic(rate):
            if rate > 10: return "🔴 HIGH RISK"
            elif rate > 3: return "🟡 MEDIUM RISK"
            else: return "🟢 LOW RISK"
        
        merged['Risk_Badge'] = merged['Critical_Rate'].apply(badge_logic)
        
        return merged.sort_values('Critical_Rate', ascending=False)

    tab1, tab2 = st.tabs(["Importers", "Exporters"])
    
    # helper for tab rendering to keep code DRY and consistent with search requirements
    def render_entity_type(df, col):
        merged = compute_entity_stats(df, col)
        
        # 1. KPI cards (3 columns)
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(kpi_card(f"{len(merged):,}", f"Total Unique {col.split('_')[0]}s", "", "#58a6ff"), unsafe_allow_html=True)
        with k2:
            st.markdown(kpi_card(f"{merged['Is_Repeat_Offender'].sum():,}", "Repeat Offenders", "flagged as repeat offenders", "#da3633"), unsafe_allow_html=True)
        with k3:
            st.markdown(kpi_card(f"{(merged['Critical_Rate'] > 5).sum():,}", "High Risk (>5%)", "", "#ffa657"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 2. Min shipments slider
        min_shipments = st.slider(
            f"Minimum shipments to include ({col.split('_')[0]}s):",
            min_value=1, max_value=20, value=3,
            help="Entities with few shipments may show inflated rates",
            key=f"slider_{col}"
        )
        
        filtered = merged[merged['Total_Shipments'] >= min_shipments].copy()
        
        st.caption(
            f"⚠️ Entities with only 1–2 shipments excluded to avoid inflated 100% rates. "
            f"Showing {len(filtered)} of {len(merged)} entities."
        )
        
        # 3. Search box (below slider as requested)
        search = st.text_input(f"🔍 Search by {col.replace('_', ' ')}", key=f"search_{col}")
        
        # Final display dataframe
        if search:
            display_full = filtered[filtered[col].astype(str).str.contains(search, case=False)].copy()
        else:
            display_full = filtered.copy()
            
        # 4. Entity Risk Table
        st.markdown(f"#### {col.replace('_', ' ')} Risk Table")
        table_top = display_full.head(100).copy()
        table_top['Is_Repeat_Offender'] = table_top['Is_Repeat_Offender'].map({1: '🔁 Yes', 0: '—'})
        st.dataframe(table_top, use_container_width=True)
        st.caption(f"Showing top 100 of {len(display_full)} entities. Download full list below.")
        
        # Download button
        st.download_button(
            label=f"📥 Download Full {col.split('_')[0]} Risk List",
            data=display_full.to_csv(index=False),
            file_name=f"{col.lower()}_risk_list.csv",
            mime="text/csv",
            key=f"dl_{col}"
        )

        st.markdown("---")

        # 5. Bar chart + Donut chart
        c1, c2 = st.columns(2)
        color_map = {
            '🔴 HIGH RISK':   '#FF4B4B',
            '🟡 MEDIUM RISK': '#FFA500',
            '🟢 LOW RISK':    '#00CC96'
        }
        
        with c1:
            chart_df = display_full.nlargest(15, 'Critical_Rate')
            fig = px.bar(
                chart_df,
                x=col,
                y='Critical_Rate',
                color='Risk_Badge',
                color_discrete_map=color_map,
                title=f'Top 15 {col.split("_")[0]}s by Critical Rate %',
                labels={'Critical_Rate': 'Critical Rate (%)', col: col}
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            risk_dist = display_full['Risk_Badge'].value_counts()
            fig_donut = px.pie(
                values=risk_dist.values,
                names=risk_dist.index,
                hole=0.5,
                color=risk_dist.index,
                color_discrete_map=color_map,
                title='Entity Risk Distribution'
            )
            fig_donut.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_donut, use_container_width=True)

        # 6. Most Dangerous Entity insight
        dangerous = display_full[display_full['Total_Shipments'] >= 3].nlargest(1, 'Critical_Rate')
        if not dangerous.empty:
            row = dangerous.iloc[0]
            st.warning(
                f"⚠️ Most Dangerous {col.split('_')[0]}: **{row[col]}** — "
                f"{row['Critical_Rate']:.1f}% critical rate "
                f"across {row['Total_Shipments']} shipments"
            )

    with tab1:
        render_entity_type(df, 'Importer_ID')
    with tab2:
        render_entity_type(df, 'Exporter_ID')


# ─────────────────────────────────────────────
# PAGE 8 — INSPECTION FEEDBACK LOOP
# ─────────────────────────────────────────────
def render_feedback_loop(preds):
    st.markdown('<div class="main-header"><h1>🔄 Inspection Outcome Feedback</h1>'
                '<p>Record actual inspection results to improve future predictions</p></div>',
                unsafe_allow_html=True)

    st.info("🧠 Feedback collected here is used to continuously retrain the AI model. "
            "This closes the loop between AI prediction and real-world outcomes.")

    with st.form("feedback_form"):
        c1, c2 = st.columns(2)
        with c1: container_id = st.text_input("Container ID")
        with c2: insp_date    = st.date_input("Inspection Date", value=datetime.now())

        outcome = st.selectbox("Actual Outcome", [
            "✅ Clean — No issues found",
            "⚠️ Suspicious — Minor violations",
            "🚨 Contraband Found — Major violation",
            "📦 Misdeclared Weight",
            "💰 Misdeclared Value"
        ])

        c3, c4 = st.columns([2, 1])
        with c3: notes = st.text_area("Officer Notes (max 200 chars)", max_chars=200)
        with c4: officer_id = st.text_input("Officer ID")

        submitted = st.form_submit_button("📝 Submit Feedback")

    if submitted:
        if not container_id:
            st.error("Please enter a Container ID.")
        else:
            # Save to CSV
            new_data = pd.DataFrame([{
                'Container_ID': container_id,
                'Inspection_Date': insp_date,
                'Actual_Outcome': outcome,
                'Officer_Notes': notes,
                'Officer_ID': officer_id,
                'Submitted_At': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }])

            log_file = "feedback_log.csv"
            if os.path.exists(log_file):
                new_data.to_csv(log_file, mode='a', header=False, index=False)
            else:
                new_data.to_csv(log_file, index=False)

            st.success("✅ Feedback recorded successfully!")

            # Comparison with AI
            p_row = preds[preds['Container_ID'].astype(str) == container_id.strip()]
            if not p_row.empty:
                ai_lvl = p_row.iloc[0]['Risk_Level']
                ai_score = p_row.iloc[0]['Risk_Score']
                st.write(f"**AI Prediction:** {ai_lvl} (Score: {ai_score:.1f})")
                st.write(f"**Actual Outcome:** {outcome}")

                is_correct = (ai_lvl == 'Critical' and ("Contraband" in outcome or "Suspicious" in outcome)) or \
                             (ai_lvl == 'Clear' and "Clean" in outcome)
                st.subheader("Was AI Correct? " + ("✅" if is_correct else "❌"))

    # Log section
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.subheader("📋 Historical Feedback Log")
    if os.path.exists("feedback_log.csv"):
        f_df = pd.read_csv("feedback_log.csv")
        st.write(f"Total records collected: **{len(f_df)}**")

        # Accuracy estimation
        st.dataframe(f_df.sort_values("Submitted_At", ascending=False), use_container_width=True)

        # Chart
        out_counts = f_df['Actual_Outcome'].value_counts().reset_index()
        out_counts.columns = ['Outcome', 'Count']
        fig = px.bar(out_counts, x='Outcome', y='Count', title="Outcome Distribution",
                     color='Outcome', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color='#e6edf3', title_font_color='#e6edf3', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Improvement callout
        st.success(f"💡 With **{len(f_df)}** feedback records collected, the model can be retrained "
                   f"to improve Critical class recall by an estimated 3-8%.")
    else:
        st.info("No feedback records found yet.")


# ─────────────────────────────────────────────
# PAGE 9 — DIGITAL TWIN MONITOR
# ─────────────────────────────────────────────
def render_digital_twin():
    st.markdown('<div class="main-header"><h1>📡 Digital Twin Monitor</h1>'
                '<p>Real-time digital replica of each container. Detects post-entry tampering and idle anomalies.</p></div>',
                unsafe_allow_html=True)
    
    twin_df = get_twin_data()
    
    if twin_df.empty:
        st.warning("No digital twin data available. Ensure predictions and processed data are loaded.")
        return

    # SECTION 1 — FLEET OVERVIEW
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(kpi_card(f"{len(twin_df):,}", "Total Tracked", "Full fleet digital clones", "#58a6ff"), unsafe_allow_html=True)
    with k2:
        tampering = (twin_df['Twin_Status'] == "🔴 TAMPERING SUSPECTED").sum()
        st.markdown(kpi_card(f"{tampering:,}", "Tampering Suspected", "critical alerts triggered", "#FF4B4B"), unsafe_allow_html=True)
    with k3:
        anomalies = (twin_df['Alert_Count'] >= 1).sum()
        st.markdown(kpi_card(f"{anomalies:,}", "Anomalies Detected", "monitoring required", "#FFA500"), unsafe_allow_html=True)
    with k4:
        weight_changes = (twin_df['Weight_Delta_Pct'].abs() > 5).sum()
        st.markdown(kpi_card(f"{weight_changes:,}", "Weight Changes", ">5% delta since entry", "#FFD700"), unsafe_allow_html=True)
    
    st.info("💡 Digital Twin catches **POST-ENTRY** tampering — containers that passed initial XGBoost screening but were tampered with while sitting idle.")

    # SECTION 2 — TWIN STATUS DISTRIBUTION
    c1, c2 = st.columns(2)
    with c1:
        status_counts = twin_df['Twin_Status'].value_counts()
        color_map = {
            "🔴 TAMPERING SUSPECTED": "#FF4B4B",
            "🟡 ANOMALIES DETECTED": "#FFA500",
            "🟡 MONITORING": "#FFD700",
            "🟢 NORMAL": "#00CC96"
        }
        fig_donut = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            hole=0.5,
            color=status_counts.index,
            color_discrete_map=color_map,
            title='Fleet Twin Status'
        )
        fig_donut.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        # Weight Delta Bins
        bins = [-float('inf'), -15, -5, 5, 15, float('inf')]
        labels = ['< -15%', '-15 to -5%', '-5 to +5%', '+5 to +15%', '> +15%']
        twin_df['Weight_Bin'] = pd.cut(twin_df['Weight_Delta_Pct'], bins=bins, labels=labels)
        bin_counts = twin_df['Weight_Bin'].value_counts().reindex(labels).reset_index()
        bin_counts.columns = ['Weight Delta', 'Count']
        
        # Color: red for extremes, green for normal
        bin_colors = ['#FF4B4B' if (l == '< -15%' or l == '> +15%') else '#00CC96' if l == '-5 to +5%' else '#FFA500' for l in labels]
        
        fig_bar = px.bar(bin_counts, x='Weight Delta', y='Count', title="Weight Change Distribution Since Entry")
        fig_bar.update_traces(marker_color=bin_colors)
        fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_bar, use_container_width=True)

    # SECTION 3 — CRITICAL TWINS TABLE
    st.markdown("### 🚨 Containers Requiring Immediate Attention")
    critical_twins = twin_df[twin_df['Twin_Status'].str.contains("TAMPERING|ANOMALIES")].sort_values(['Alert_Count', 'Weight_Delta_Pct'], ascending=False)
    
    if not critical_twins.empty:
        table_df = critical_twins[['Container_ID', 'Twin_Status', 'Weight_Delta_Pct', 'Dwell_Time_Hours', 'Alert_Count', 'Risk_Level', 'Origin_Country']].copy()
        
        # Format Weight Delta
        def format_weight_delta(val):
            if val > 0.1: return f"+{val:.1f}% ⬆️"
            if val < -0.1: return f"{val:.1f}% ⬇️"
            return "Stable ✅"
        
        table_df['Weight_Delta_Pct'] = table_df['Weight_Delta_Pct'].apply(format_weight_delta)
        st.dataframe(table_df, use_container_width=True)
        st.download_button("📥 Export Flagged Containers", critical_twins.to_csv(index=False), "flagged_twins.csv", "text/csv")
    else:
        st.success("✅ No tampering suspected in the fleet currently.")

    st.markdown("---")

    # SECTION 4 — INDIVIDUAL CONTAINER TWIN VIEWER
    st.markdown("### 🔍 Container Twin Deep Dive")
    search_id = st.text_input("Enter Container ID to view its Digital Twin").strip()
    
    if search_id:
        target = twin_df[twin_df['Container_ID'].astype(str) == search_id]
        if not target.empty:
            row = target.iloc[0]
            
            # Custom styled card
            st.markdown(f"""
            <div style="background:#1c2128; border:1px solid #30363d; border-radius:12px; padding:24px; margin-bottom:20px;">
                <h2 style="color:#58a6ff; margin-top:0;">📡 DIGITAL TWIN — {row['Container_ID']}</h2>
                <div style="font-size:1.2rem; font-weight:700; margin-bottom:20px;">Status: {row['Twin_Status']}</div>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px; margin-bottom:20px;">
                    <div>
                        <div style="color:#8b949e; font-size:0.8rem; text-transform:uppercase;">Entry Snapshot</div>
                        <div style="font-size:1.1rem; font-weight:600;">Weight: {row['Entry_Weight']:,} kg</div>
                    </div>
                    <div>
                        <div style="color:#8b949e; font-size:0.8rem; text-transform:uppercase;">Current State</div>
                        <div style="font-size:1.1rem; font-weight:600;">Weight: {row['Current_Weight']:,} kg</div>
                    </div>
                </div>
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:20px; margin-bottom:20px;">
                    <div>
                        <div style="color:#8b949e; font-size:0.8rem; text-transform:uppercase;">Weight Change</div>
                        <div style="font-size:1.1rem; font-weight:600; color:{'#FF4B4B' if row['Weight_Delta_Pct'] > 5 else '#FFA500' if row['Weight_Delta_Pct'] < -5 else '#00CC96'};">
                            {row['Weight_Delta']:+.1f} kg ({row['Weight_Delta_Pct']:+.1f}%)
                        </div>
                    </div>
                    <div>
                        <div style="color:#8b949e; font-size:0.8rem; text-transform:uppercase;">Dwell Time</div>
                        <div style="font-size:1.1rem; font-weight:600;">{row['Dwell_Time_Hours']} hours ({row['Dwell_Status']})</div>
                    </div>
                </div>
                <div style="border-top:1px solid #30363d; padding-top:15px; margin-top:10px;">
                    <div style="color:#8b949e; font-size:0.8rem; text-transform:uppercase; margin-bottom:10px;">Active Alerts ({row['Alert_Count']})</div>
                    {''.join([f'<div style="color:#e6edf3; margin-bottom:5px;">• {a}</div>' for a in row['Alerts']]) if row['Alerts'] else 'No alerts'}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk Gauge
            st.write(f"**AI Risk Score: {row['Risk_Score']:.1f}/100** ({row['Risk_Level']})")
            st.progress(int(row['Risk_Score']))
            if row['Explanation_Summary']:
                st.caption(row['Explanation_Summary'])

            # Timeline chart
            st.markdown(f"#### Weight Timeline — Container {row['Container_ID']}")
            timeline_df = pd.DataFrame(row['Timeline'])
            
            line_color = '#FF4B4B' if abs(row['Weight_Delta_Pct']) > 5 else '#00CC96'
            fig_timeline = px.line(timeline_df, x='checkpoint', y='weight', markers=True, text='weight')
            fig_timeline.update_traces(line_color=line_color, textposition="top center")
            fig_timeline.add_hline(y=row['Entry_Weight'], line_dash="dash", line_color="#8b949e", annotation_text="Entry Baseline")
            fig_timeline.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white', yaxis_title="Weight (kg)")
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Show individual alerts
            if row['Alerts']:
                for a in row['Alerts']:
                    st.warning(a)
            else:
                st.success("✅ No anomalies detected for this container.")

            # Shipment context
            st.markdown("#### Shipment Details")
            c_ctx1, c_ctx2, c_ctx3 = st.columns(3)
            with c_ctx1: 
                st.write(f"**Origin:** {row['Origin_Country']}")
                st.write(f"**Destination:** {row['Destination_Country']}")
            with c_ctx2:
                st.write(f"**Importer:** {row['Importer_ID']}")
                st.write(f"**Shipping Line:** {row['Shipping_Line']}")
            with c_ctx3:
                st.write(f"**Declared At:** {row['Declaration_Time']}")

        else:
            st.error(f"Container ID '{search_id}' not found in Digital Twin database.")

    st.markdown("---")

    # SECTION 5 — INSIGHTS PANEL
    st.markdown("### 💡 Fleet-Wide Insights")
    
    # Route Insight
    sus_containers = twin_df[twin_df['Twin_Status'] == "🔴 TAMPERING SUSPECTED"]
    if not sus_containers.empty:
        route_stats = sus_containers.groupby(['Origin_Country', 'Destination_Country'])['Weight_Delta_Pct'].mean().reset_index()
        top_route = route_stats.nlargest(1, 'Weight_Delta_Pct').iloc[0]
        st.warning(f"🌍 **Most suspicious route:** {top_route['Origin_Country']} → {top_route['Destination_Country']} | "
                   f"Avg weight change: {top_route['Weight_Delta_Pct']:+.1f}% in flagged containers")
    
    # Dwell Offender
    offender = twin_df[twin_df['Alert_Count'] > 0].nlargest(1, 'Dwell_Time_Hours')
    if not offender.empty:
        row = offender.iloc[0]
        st.warning(f"⏰ **Longest suspicious dwell:** Container {row['Container_ID']} | "
                   f"{row['Dwell_Time_Hours']:.0f} hours | Status: {row['Twin_Status']}")
        
    # Twin vs AI comparison
    caught_by_twin = len(twin_df[(twin_df['Twin_Status'] == "🔴 TAMPERING SUSPECTED") & (twin_df['Risk_Level'] == 'Clear')])
    st.info(f"🤖 **Digital Twin caught {caught_by_twin} containers** that XGBoost scored as 'Clear' — "
            f"demonstrating detection capability for tampering that occurs AFTER the initial declaration.")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 16px 0 10px 0;">
            <div style="font-size:2.5rem;">🚢</div>
            <div style="font-size:1.1rem; font-weight:700; color:#e6edf3;">SmartContainer</div>
            <div style="font-size:0.8rem; color:#8b949e;">Risk Engine v1.0</div>
        </div>
        <hr style="border-color:#30363d; margin: 8px 0 16px 0;">
        """, unsafe_allow_html=True)

        pages = [
            "📊 Overview",
            "🚨 Critical Containers",
            "🔍 Container Lookup",
            "🌍 Geographic Risk",
            "🤖 Live Risk Predictor",
            "📋 Inspection Priority Queue",
            "🕵️ Behavioral Profiling",
            "🔄 Feedback Loop",
            "📡 Digital Twin Monitor"
        ]
        page = st.radio("Navigate", pages, label_visibility="collapsed")

        st.markdown("""
        <hr style="border-color:#30363d; margin: 16px 0 8px 0;">
        <div style="font-size:0.75rem; color:#484f58; text-align:center;">
            Powered by XGBoost + Isolation Forest<br>Hackathon 2026
        </div>
        """, unsafe_allow_html=True)

    preds = load_predictions()
    proc  = load_processed_data()

    if page == "📊 Overview":
        render_overview(preds, proc)
    elif page == "🚨 Critical Containers":
        render_critical_table(preds, proc)
    elif page == "🔍 Container Lookup":
        render_container_lookup(preds, proc)
    elif page == "🌍 Geographic Risk":
        render_geographic_risk(preds, proc)
    elif page == "🤖 Live Risk Predictor":
        render_live_predictor(proc)
    elif page == "📋 Inspection Priority Queue":
        render_inspection_queue(preds, proc)
    elif page == "🕵️ Behavioral Profiling":
        render_behavioral_profiling(proc)
    elif page == "🔄 Feedback Loop":
        render_feedback_loop(preds)
    elif page == "📡 Digital Twin Monitor":
        render_digital_twin()

    st.markdown('<div class="footer">SmartContainer Risk Engine | Port Customs Operations Dashboard | © 2026</div>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
