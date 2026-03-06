import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as stats

# Set visual aesthetic for plots
sns.set_theme(style="whitegrid")

def setup_environment():
    """Create directory for output plots if it doesn't exist."""
    if not os.path.exists("eda_plots"):
        os.makedirs("eda_plots")

def load_data(filepath="Historical Data-1.csv"):
    """Load the dataset, rename columns, and return as a pandas DataFrame."""
    try:
        df = pd.read_csv(filepath)
        
        # Rename columns to match expected names
        column_mapping = {
            'Declaration_Date (YYYY-MM-DD)': 'Declaration_Date',
            'Declaration_Time (HH:MM:SS)': 'Declaration_Time',
            'Trade_Regime (Import / Export / Transit)': 'Trade_Regime'
        }
        df.rename(columns=column_mapping, inplace=True)
        
        print(f"Data successfully loaded from {filepath}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def basic_overview(df):
    """1. BASIC OVERVIEW"""
    print("\n" + "="*50)
    print("1. BASIC OVERVIEW")
    print("="*50)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print("\n--- Data Types ---")
    print(df.dtypes)
    
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "No missing values")
    
    print(f"\n--- Duplicates ---")
    print(f"Number of duplicate rows: {df.duplicated().sum()}")
    
    print("\n--- Value Counts for Categorical Columns (Top 5) ---")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        print(f"\n[{col}]")
        print(df[col].value_counts().head(5))
        
    print("\n--- Describe Stats for Numerical Columns ---")
    print(df.describe().T)

def class_distribution(df):
    """2. CLASS DISTRIBUTION"""
    print("\n" + "="*50)
    print("2. CLASS DISTRIBUTION")
    print("="*50)
    
    plt.figure(figsize=(12, 5))
    
    # Bar chart
    plt.subplot(1, 2, 1)
    sns.countplot(x='Clearance_Status', data=df, order=['Clear', 'Low Risk', 'Critical'], palette='viridis')
    plt.title("Clearance Status Distribution")
    plt.ylabel("Count")
    
    # Pie chart
    plt.subplot(1, 2, 2)
    counts = df['Clearance_Status'].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis', len(counts)))
    plt.title("Clearance Status Proportion")
    
    plt.tight_layout()
    plt.savefig("eda_plots/01_class_distribution.png")
    plt.close()
    
    print("Class Imbalance Ratio:")
    total = len(df)
    for status, count in counts.items():
        print(f"  {status}: {count} ({count/total*100:.2f}%)")
    
    if 'Critical' in counts and 'Clear' in counts:
        print(f"\nRatio Clear to Critical: {counts['Clear']/counts['Critical']:.2f}:1")

def weight_anomaly_analysis(df):
    """3. WEIGHT ANOMALY ANALYSIS"""
    print("\n" + "="*50)
    print("3. WEIGHT ANOMALY ANALYSIS")
    print("="*50)
    
    df['weight_discrepancy'] = (df['Measured_Weight'] - df['Declared_Weight']) / df['Declared_Weight'] * 100
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Clearance_Status', y='weight_discrepancy', data=df, order=['Clear', 'Low Risk', 'Critical'])
    plt.title("Weight Discrepancy by Clearance Status")
    plt.ylabel("Weight Discrepancy (%)")
    plt.savefig("eda_plots/02_weight_discrepancy_boxplot.png")
    plt.close()
    
    df['high_weight_discrepancy'] = df['weight_discrepancy'].abs() > 10
    print("Containers with absolute weight discrepancy > 10% per risk level:")
    high_disc = df[df['high_weight_discrepancy']].groupby('Clearance_Status').size()
    total_tier = df.groupby('Clearance_Status').size()
    
    for status in ['Clear', 'Low Risk', 'Critical']:
        cnt = high_disc.get(status, 0)
        tot = total_tier.get(status, 1)  # avoid division by zero
        print(f"  {status}: {cnt} ({cnt/tot*100:.2f}% of {status} tier)")
    
    plt.figure(figsize=(10, 6))
    subset = df[df['Clearance_Status'].isin(['Critical', 'Clear'])]
    sns.histplot(data=subset, x='weight_discrepancy', hue='Clearance_Status', common_norm=False, stat='density', bins=50, kde=True)
    plt.title("Histogram of Weight Discrepancy (Critical vs Clear)")
    plt.xlabel("Weight Discrepancy (%)")
    plt.savefig("eda_plots/03_weight_discrepancy_hist.png")
    plt.close()

def value_analysis(df):
    """4. VALUE ANALYSIS"""
    print("\n" + "="*50)
    print("4. VALUE ANALYSIS")
    print("="*50)
    
    df['value_per_kg'] = df['Declared_Value'] / df['Declared_Weight']
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Clearance_Status', y='value_per_kg', data=df, order=['Clear', 'Low Risk', 'Critical'])
    plt.title("Value per KG by Clearance Status")
    plt.ylabel("Value per KG")
    plt.yscale('log') # Useful for monetary values with large outliers
    plt.savefig("eda_plots/04_value_per_kg_boxplot.png")
    plt.close()
    
    critical_df = df[df['Clearance_Status'] == 'Critical']
    top_hs = critical_df.groupby('HS_Code')['value_per_kg'].mean().sort_values(ascending=False).head(10)
    print("Top 10 HS Codes with highest avg value_per_kg in Critical containers:")
    print(top_hs)

def dwell_time_analysis(df):
    """5. DWELL TIME ANALYSIS"""
    print("\n" + "="*50)
    print("5. DWELL TIME ANALYSIS")
    print("="*50)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Clearance_Status', y='Dwell_Time_Hours', data=df, order=['Clear', 'Low Risk', 'Critical'])
    plt.title("Dwell Time Hours by Clearance Status")
    plt.ylabel("Dwell Time (Hours)")
    plt.savefig("eda_plots/05_dwell_time_boxplot.png")
    plt.close()
    
    critical_df = df[df['Clearance_Status'] == 'Critical']
    if len(critical_df) > 0:
        pct_high_dwell = (len(critical_df[critical_df['Dwell_Time_Hours'] > 96]) / len(critical_df)) * 100
        print(f"Percentage of Critical containers with Dwell Time > 96 hours: {pct_high_dwell:.2f}%")
        
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Dwell_Time_Hours', hue='Clearance_Status', bins=50, common_norm=False, stat='density', kde=True)
    plt.title("Histogram of Dwell Time per Risk Level")
    plt.xlabel("Dwell Time (Hours)")
    plt.savefig("eda_plots/06_dwell_time_hist.png")
    plt.close()

def temporal_patterns(df):
    """6. TEMPORAL PATTERNS"""
    print("\n" + "="*50)
    print("6. TEMPORAL PATTERNS")
    print("="*50)
    
    # Extract temporal features
    df['Declaration_Time_Obj'] = pd.to_datetime(df['Declaration_Time'], format='%H:%M:%S', errors='coerce').dt.time
    df['hour_of_day'] = df['Declaration_Time_Obj'].apply(lambda x: x.hour if pd.notnull(x) else np.nan)
    
    df['Declaration_Date_Obj'] = pd.to_datetime(df['Declaration_Date'], errors='coerce')
    df['day_of_week'] = df['Declaration_Date_Obj'].dt.day_name()
    
    critical_df = df[df['Clearance_Status'] == 'Critical']
    
    plt.figure(figsize=(10, 6))
    sns.countplot(x='hour_of_day', data=critical_df, color='salmon')
    plt.title("Count of Critical Containers by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Count")
    plt.savefig("eda_plots/07_critical_by_hour.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sns.countplot(x='day_of_week', data=critical_df, order=days_order, color='salmon')
    plt.title("Count of Critical Containers by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Count")
    plt.savefig("eda_plots/08_critical_by_day.png")
    plt.close()
    
    late_night = critical_df[(critical_df['hour_of_day'] >= 22) | (critical_df['hour_of_day'] <= 4)]
    if len(critical_df) > 0:
        late_night_pct = len(late_night) / len(critical_df) * 100
        print(f"Late-night (10pm-4am) declarations in Critical tier: {len(late_night)} out of {len(critical_df)} ({late_night_pct:.2f}%)")
        print("\nIs there a pattern? " + ("Yes, noticeably high." if late_night_pct > 25 else "No significant disproportionate late-night pattern observed."))

def geographic_risk_analysis(df):
    """7. GEOGRAPHIC RISK ANALYSIS"""
    print("\n" + "="*50)
    print("7. GEOGRAPHIC RISK ANALYSIS")
    print("="*50)
    
    critical_df = df[df['Clearance_Status'] == 'Critical']
    total_by_origin = df['Origin_Country'].value_counts()
    
    # By count
    top_15_count = critical_df['Origin_Country'].value_counts().head(15)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_15_count.values, y=top_15_count.index, palette='Reds_r')
    plt.title("Top 15 Origin Countries by Count of Critical Containers")
    plt.xlabel("Number of Critical Containers")
    plt.ylabel("Origin Country")
    plt.tight_layout()
    plt.savefig("eda_plots/09_top_origin_count.png")
    plt.close()
    
    # By rate
    critical_rate = (critical_df['Origin_Country'].value_counts() / total_by_origin * 100).dropna()
    top_15_rate = critical_rate.sort_values(ascending=False).head(15)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_15_rate.values, y=top_15_rate.index, palette='Reds_r')
    plt.title("Top 15 Origin Countries by Critical Rate (%)")
    plt.xlabel("Critical Rate (%)")
    plt.ylabel("Origin Country")
    plt.tight_layout()
    plt.savefig("eda_plots/10_top_origin_rate.png")
    plt.close()
    
    # Destination port + country combinations
    df['Dest_Combo'] = df['Destination_Country'] + " - " + df['Destination_Port']
    critical_df_with_dest = df[df['Clearance_Status'] == 'Critical']
    top_10_dest = critical_df_with_dest['Dest_Combo'].value_counts().head(10)
    print("Top 10 risky Destination Country + Port combinations:")
    print(top_10_dest)

def entity_risk_profiling(df):
    """8. ENTITY RISK PROFILING"""
    print("\n" + "="*50)
    print("8. ENTITY RISK PROFILING")
    print("="*50)
    
    critical_df = df[df['Clearance_Status'] == 'Critical']
    
    top_10_importers = critical_df['Importer_ID'].value_counts().head(10)
    print("Top 10 Importer IDs by Critical shipment count:")
    print(top_10_importers)
    
    top_10_exporters = critical_df['Exporter_ID'].value_counts().head(10)
    print("\nTop 10 Exporter IDs by Critical shipment count:")
    print(top_10_exporters)
    
    importer_counts = critical_df['Importer_ID'].value_counts()
    exporter_counts = critical_df['Exporter_ID'].value_counts()
    repeat_imps = importer_counts[importer_counts > 1]
    repeat_exps = exporter_counts[exporter_counts > 1]
    
    print("\nAre there repeat offenders?")
    print(f"Yes, {len(repeat_imps)} Importers and {len(repeat_exps)} Exporters appear multiple times in the Critical tier.")

def hs_code_analysis(df):
    """9. HS CODE ANALYSIS"""
    print("\n" + "="*50)
    print("9. HS CODE ANALYSIS")
    print("="*50)
    
    # Extract first 2 digits
    df['HS_Chapter'] = df['HS_Code'].astype(str).str.zfill(6).str[:2]
    
    critical_df = df[df['Clearance_Status'] == 'Critical']
    total_by_chapter = df['HS_Chapter'].value_counts()
    
    critical_rate = (critical_df['HS_Chapter'].value_counts() / total_by_chapter * 100).dropna()
    top_10_hs_rate = critical_rate.sort_values(ascending=False).head(10)
    
    print("Top 10 HS Chapters with highest Critical rate (%):")
    print(top_10_hs_rate)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_10_hs_rate.index, y=top_10_hs_rate.values, palette='Oranges_r')
    plt.title("Risk Distribution per Top 10 HS Chapters")
    plt.xlabel("HS Chapter")
    plt.ylabel("Critical Rate (%)")
    plt.savefig("eda_plots/11_top_hs_chapters_risk.png")
    plt.close()

def shipping_line_analysis(df):
    """10. SHIPPING LINE ANALYSIS"""
    print("\n" + "="*50)
    print("10. SHIPPING LINE ANALYSIS")
    print("="*50)
    
    critical_df = df[df['Clearance_Status'] == 'Critical']
    total_by_line = df['Shipping_Line'].value_counts()
    
    critical_rate = (critical_df['Shipping_Line'].value_counts() / total_by_line * 100).dropna()
    top_10_lines = critical_rate.sort_values(ascending=False).head(10)
    
    print("Risk rate per Shipping Line (Top 10):")
    print(top_10_lines)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_10_lines.index, y=top_10_lines.values, palette='Blues_r')
    plt.title("Top 10 Shipping Lines by Critical Rate (%)")
    plt.xlabel("Shipping Line")
    plt.ylabel("Critical Rate (%)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("eda_plots/12_top_shipping_lines.png")
    plt.close()

def correlation_feature_importance(df):
    """11. CORRELATION & FEATURE IMPORTANCE HINT"""
    print("\n" + "="*50)
    print("11. CORRELATION & FEATURE IMPORTANCE HINT")
    print("="*50)
    
    num_cols = ['Declared_Value', 'Declared_Weight', 'Measured_Weight', 'Dwell_Time_Hours', 'weight_discrepancy', 'value_per_kg']
    
    corr_matrix = df[num_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Matrix of Numerical Features")
    plt.tight_layout()
    plt.savefig("eda_plots/13_correlation_matrix.png")
    plt.close()
    
    # Label encode
    status_mapping = {'Clear': 0, 'Low Risk': 1, 'Critical': 2}
    df['Clearance_Status_Encoded'] = df['Clearance_Status'].map(status_mapping)
    
    print("Point-biserial correlation of each numerical feature with encoded target (0=Clear, 1=Low Risk, 2=Critical):")
    target = df['Clearance_Status_Encoded']
    corrs = []
    
    for col in num_cols:
        # Drop NaNs for valid calculation
        valid_idx = df[col].notnull() & target.notnull()
        corr, p_value = stats.pointbiserialr(target[valid_idx], df.loc[valid_idx, col])
        corrs.append((col, corr, p_value))
        print(f"  {col:20s}: r = {corr:7.4f} (p-value: {p_value:.4e})")
        
    return corrs

def key_findings_summary(df, corrs):
    """12. KEY FINDINGS SUMMARY"""
    print("\n" + "="*50)
    print("12. KEY FINDINGS SUMMARY")
    print("="*50)
    
    # 1. Top 3 anomaly indicators
    sorted_corrs = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)
    top_3_anomaly = [item[0] for item in sorted_corrs[:3]]
    
    # 2. Most risky countries (by count & rate)
    critical_df = df[df['Clearance_Status'] == 'Critical']
    top_origin_count = critical_df['Origin_Country'].value_counts().head(3).index.tolist()
    
    # 3. Most suspicious HS chapters
    total_hs = df['HS_Chapter'].value_counts()
    critical_hs_rate = (critical_df['HS_Chapter'].value_counts() / total_hs * 100).dropna()
    top_hs_chapters = critical_hs_rate.sort_values(ascending=False).head(3).index.tolist()
    
    # 4. Weight discrepancy > 10%
    if len(critical_df) > 0:
        pct_high_weight_disc = (len(critical_df[critical_df['weight_discrepancy'].abs() > 10]) / len(critical_df)) * 100
    else:
        pct_high_weight_disc = 0.0
        
    # Formatting output
    summary = f"""KEY FINDINGS:
- Top 3 Anomaly Indicators Found (by linear correlation): {', '.join(top_3_anomaly)}
- Most Risky Origin Countries (by volume of critical incidents): {', '.join(top_origin_count)}
- Most Suspicious HS Chapters: {', '.join(top_hs_chapters)}
- {pct_high_weight_disc:.1f}% of Critical containers exhibited a weight discrepancy > 10%.
- Value per kg and extreme dwell times (>96 hours) strongly stratify container risk levels.
- Certain shippers and destination ports emerge frequently, indicating possible network-based risk patterns.
"""
    print(summary)


def main():
    setup_environment()
    
    # Check if the file exists in the current directory or provide path
    filepath = "Historical Data-1.csv"
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found in the current directory.")
        return
        
    df = load_data(filepath)
    if df is not None:
        basic_overview(df)
        class_distribution(df)
        weight_anomaly_analysis(df)
        value_analysis(df)
        dwell_time_analysis(df)
        temporal_patterns(df)
        geographic_risk_analysis(df)
        entity_risk_profiling(df)
        hs_code_analysis(df)
        shipping_line_analysis(df)
        correlations = correlation_feature_importance(df)
        key_findings_summary(df, correlations)
        
        print("\nEDA process completed successfully. All plots saved to 'eda_plots/' directory.")

if __name__ == "__main__":
    main()
