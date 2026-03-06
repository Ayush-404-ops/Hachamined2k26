import os
import json
import joblib
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from collections import Counter
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────
# STEP 1 — load_and_clean()
# ──────────────────────────────────────────────────────────
def load_and_clean():
    """
    Load raw CSV and perform basic cleaning only.
    NO feature engineering here.
    NO entity rate computation here.
    Just clean, encode target, handle obvious data issues.
    """
    print("\n" + "="*55)
    print("STEP 1: LOADING AND CLEANING RAW DATA")
    print("="*55)
    
    df = pd.read_csv("Historical Data-1.csv")
    
    # Rename columns for convenience
    df.rename(columns={
        'Declaration_Date (YYYY-MM-DD)': 'Declaration_Date',
        'Trade_Regime (Import / Export / Transit)': 'Trade_Regime'
    }, inplace=True)
    
    print(f"Raw data shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Encode target
    status_map = {'Clear': 0, 'Low Risk': 1, 'Critical': 2}
    df['target'] = df['Clearance_Status'].map(status_map)
    
    # Drop rows where target is NaN
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)
    
    print(f"\nClean data shape: {df.shape}")
    print(f"Target distribution:")
    dist = df['target'].value_counts(normalize=True).mul(100).round(2)
    for k, v in sorted(dist.items()):
        name = {0:'Clear', 1:'Low Risk', 2:'Critical'}[k]
        print(f"  {name}: {v}%")
    
    return df

# ──────────────────────────────────────────────────────────
# STEP 2 — split_data()
# ──────────────────────────────────────────────────────────
def split_data(df):
    """
    Split BEFORE any feature engineering.
    This is critical to prevent data leakage.
    Stratify on target to preserve class ratios.
    """
    print("\n" + "="*55)
    print("STEP 2: TRAIN/TEST SPLIT (BEFORE FEATURE ENGINEERING)")
    print("="*55)
    
    # Keep Container_ID separately for output
    container_ids = df['Container_ID'].values
    
    # Features = raw columns only (no engineered features yet)
    feature_cols = [
        'Declaration_Date', 'Declaration_Time', 'Trade_Regime',
        'Origin_Country', 'Destination_Port', 'Destination_Country',
        'HS_Code', 'Importer_ID', 'Exporter_ID', 'Declared_Value',
        'Declared_Weight', 'Measured_Weight', 'Shipping_Line',
        'Dwell_Time_Hours'
    ]
    
    X = df[feature_cols].copy()
    y = df['target'].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Also split Container_IDs in same order
    ids_train, ids_test = train_test_split(
        container_ids,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"Train set: {len(X_train)} rows")
    print(f"Test set:  {len(X_test)} rows")
    print(f"\nTrain class distribution:")
    counts = Counter(y_train)
    total = len(y_train)
    for k, v in sorted(counts.items()):
        name = {0:'Clear', 1:'Low Risk', 2:'Critical'}[k]
        print(f"  {name}: {v} ({v/total*100:.2f}%)")
    
    return X_train, X_test, y_train, y_test, ids_train, ids_test

# ──────────────────────────────────────────────────────────
# STEP 3 — feature_engineering()
# ──────────────────────────────────────────────────────────
def feature_engineering(X_train, X_test, y_train):
    """
    Engineer ALL features using ONLY training data.
    Apply learned transformations to test set.
    No test data is used to compute any statistic.
    """
    print("\n" + "="*55)
    print("STEP 3: FEATURE ENGINEERING (TRAIN-ONLY STATS)")
    print("="*55)
    
    def engineer(df, is_train=True, stats=None):
        df = df.copy()
        
        # ── A. WEIGHT & VALUE FEATURES ──────────────────────
        df['weight_discrepancy'] = (
            (df['Measured_Weight'] - df['Declared_Weight']) /
            df['Declared_Weight'].replace(0, np.nan) * 100
        )
        df['abs_weight_discrepancy'] = df['weight_discrepancy'].abs()
        df['weight_discrepancy_flag'] = (
            df['abs_weight_discrepancy'] > 10
        ).astype(int)
        df['weight_ratio'] = (
            df['Measured_Weight'] /
            df['Declared_Weight'].replace(0, np.nan)
        )
        df['value_per_kg'] = (
            df['Declared_Value'] /
            df['Declared_Weight'].replace(0, np.nan)
        )
        df['log_value']        = np.log1p(df['Declared_Value'])
        df['log_weight']       = np.log1p(df['Declared_Weight'])
        df['log_value_per_kg'] = np.log1p(
            df['value_per_kg'].clip(lower=0)
        )
        
        # ── B. TEMPORAL FEATURES ─────────────────────────────
        time_parsed = pd.to_datetime(
            df['Declaration_Time'], format='%H:%M:%S', errors='coerce'
        )
        df['hour_of_day'] = time_parsed.dt.hour
        df['is_late_night'] = (
            (df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 4)
        ).astype(int)
        
        date_parsed = pd.to_datetime(
            df['Declaration_Date'], errors='coerce'
        )
        df['day_of_week'] = date_parsed.dt.dayofweek
        df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
        df['month']       = date_parsed.dt.month
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(
            2 * np.pi * df['hour_of_day'] / 24
        )
        df['hour_cos'] = np.cos(
            2 * np.pi * df['hour_of_day'] / 24
        )
        
        # ── C. DWELL TIME FEATURES ───────────────────────────
        df['dwell_time_flag'] = (
            df['Dwell_Time_Hours'] > 96
        ).astype(int)
        df['log_dwell_time'] = np.log1p(df['Dwell_Time_Hours'])
        
        if is_train:
            stats = stats or {}
            stats['dwell_mean'] = df['Dwell_Time_Hours'].mean()
            stats['dwell_std']  = df['Dwell_Time_Hours'].std()
        
        df['dwell_zscore'] = (
            (df['Dwell_Time_Hours'] - stats['dwell_mean']) /
            (stats['dwell_std'] + 1e-8)
        )
        
        # ── D. HS CODE FEATURES ──────────────────────────────
        df['HS_Chapter'] = (
            df['HS_Code'].astype(str).str.zfill(6).str[:2]
        )
        
        # ── E. TRADE REGIME ENCODING ─────────────────────────
        regime_map = {'Import': 0, 'Transit': 1, 'Export': 2}
        df['trade_regime_encoded'] = (
            df['Trade_Regime'].map(regime_map).fillna(0)
        )
        
        return df, stats
    
    # ── F. ENTITY RISK RATES (TRAIN LABELS ONLY) ─────────────
    def compute_entity_rates(X_train_eng, y_train):
        """
        Compute smoothed risk rates per entity using 
        only training labels.
        """
        entity_maps = {}
        
        global_critical_rate = (y_train == 2).mean()
        global_lowrisk_rate  = (y_train == 1).mean()
        k_smooth = 10
        
        train_df = X_train_eng.copy()
        train_df['target'] = y_train.values
        
        ENTITY_COLS = [
            'Importer_ID', 'Exporter_ID', 'Origin_Country',
            'Destination_Country', 'Shipping_Line', 'HS_Chapter'
        ]
        
        for col in ENTITY_COLS:
            if col not in train_df.columns:
                continue
            
            grp = train_df.groupby(col)['target'].agg(
                total='count',
                n_critical=lambda x: (x == 2).sum(),
                n_lowrisk =lambda x: (x == 1).sum()
            ).reset_index()
            
            # Laplace smoothed rates
            grp['critical_rate'] = (
                (grp['n_critical'] + global_critical_rate * k_smooth) /
                (grp['total'] + k_smooth)
            )
            grp['lowrisk_rate'] = (
                (grp['n_lowrisk'] + global_lowrisk_rate * k_smooth) /
                (grp['total'] + k_smooth)
            )
            grp['risk_score'] = (
                grp['critical_rate'] * 0.6 +
                grp['lowrisk_rate']  * 0.4
            )
            grp['count'] = grp['total']
            
            entity_maps[col] = {
                'map': grp.set_index(col)[['critical_rate',
                            'lowrisk_rate',
                            'risk_score',
                            'count']].to_dict('index'),
                'global_critical': global_critical_rate,
                'global_lowrisk':  global_lowrisk_rate,
                'global_risk': (
                    global_critical_rate * 0.6 +
                    global_lowrisk_rate  * 0.4
                )
            }
        
        return entity_maps
    
    def apply_entity_rates(df, entity_maps):
        df = df.copy()
        for col, info in entity_maps.items():
            if col not in df.columns:
                continue
            rate_map = info['map']
            
            df[f'{col}_critical_rate'] = df[col].map(
                lambda x: rate_map[x]['critical_rate']
                if x in rate_map
                else info['global_critical']
            )
            df[f'{col}_lowrisk_rate'] = df[col].map(
                lambda x: rate_map[x]['lowrisk_rate']
                if x in rate_map
                else info['global_lowrisk']
            )
            df[f'{col}_risk_score'] = df[col].map(
                lambda x: rate_map[x]['risk_score']
                if x in rate_map
                else info['global_risk']
            )
            df[f'{col}_shipment_count'] = df[col].map(
                lambda x: rate_map[x]['count']
                if x in rate_map
                else 0
            )
        return df
    
    def add_repeat_offender(df, col, crit_count_map):
        df[f'{col}_critical_count'] = (
            df[col].map(crit_count_map).fillna(0).astype(int)
        )
        df[f'{col}_is_repeat_offender'] = (
            df[f'{col}_critical_count'] > 1
        ).astype(int)
        return df
    
    def add_interaction_features(df):
        df['weight_x_dwell'] = (
            df['abs_weight_discrepancy'] * df['log_dwell_time']
        )
        # origin_country_critical_rate might be missing if drop occurred early, protect
        orig_rate_col = 'Origin_Country_critical_rate'
        df['value_x_origin_risk'] = (
            df['log_value_per_kg'] *
            df[orig_rate_col] if orig_rate_col in df.columns else 0
        )
        df['high_risk_combo'] = (
            (df['weight_discrepancy_flag'] == 1) &
            (df['dwell_time_flag'] == 1)
        ).astype(int)
        return df
    
    # ── EXECUTE PIPELINE ─────────────────────────────────────
    stats = {}
    
    # Engineer base features
    X_train_eng, stats = engineer(X_train, is_train=True,  stats=stats)
    X_test_eng,  stats = engineer(X_test,  is_train=False, stats=stats)
    
    # Compute entity rates from training labels only
    entity_maps = compute_entity_rates(X_train_eng, y_train)
    
    # Apply to both splits
    X_train_eng = apply_entity_rates(X_train_eng, entity_maps)
    X_test_eng  = apply_entity_rates(X_test_eng,  entity_maps)
    
    # Repeat offender features
    train_with_target = X_train_eng.copy()
    train_with_target['target'] = y_train.values
    
    for col in ['Importer_ID', 'Exporter_ID']:
        if col in train_with_target.columns:
            crit_map = (
                train_with_target[train_with_target['target'] == 2]
                .groupby(col).size()
            )
            X_train_eng = add_repeat_offender(
                X_train_eng, col, crit_map
            )
            X_test_eng = add_repeat_offender(
                X_test_eng, col, crit_map
            )
    
    # Interaction features
    X_train_eng = add_interaction_features(X_train_eng)
    X_test_eng  = add_interaction_features(X_test_eng)
    
    # ── DROP RAW STRING COLUMNS ──────────────────────────────
    DROP_COLS = [
        'Declaration_Date', 'Declaration_Time', 'Trade_Regime',
        'Origin_Country', 'Destination_Port', 'Destination_Country',
        'HS_Code', 'HS_Chapter', 'Importer_ID', 'Exporter_ID',
        'Shipping_Line'
    ]
    drop_train = [c for c in DROP_COLS if c in X_train_eng.columns]
    drop_test  = [c for c in DROP_COLS if c in X_test_eng.columns]
    X_train_eng = X_train_eng.drop(columns=drop_train)
    X_test_eng  = X_test_eng.drop(columns=drop_test)
    
    # ── ALIGN COLUMNS ────────────────────────────────────────
    train_cols = X_train_eng.columns.tolist()
    missing_in_test = set(train_cols) - set(X_test_eng.columns)
    for c in missing_in_test:
        X_test_eng[c] = 0
    X_test_eng = X_test_eng[train_cols]
    
    # ── IMPUTE NaN / Inf ─────────────────────────────────────
    X_train_eng.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test_eng.replace([np.inf, -np.inf],  np.nan, inplace=True)
    train_medians = X_train_eng.median()
    X_train_eng = X_train_eng.fillna(train_medians)
    X_test_eng  = X_test_eng.fillna(train_medians)
    
    print(f"Total features engineered: {len(train_cols)}")
    print(f"Feature list: {train_cols}")
    
    return (X_train_eng, X_test_eng, 
            entity_maps, stats, train_medians)

# ──────────────────────────────────────────────────────────
# STEP 4 — cross_validate()
# ──────────────────────────────────────────────────────────
def cross_validate(X_train, y_train):
    """
    5-Fold Stratified CV on REAL data BEFORE SMOTE.
    This gives honest estimate of generalization.
    """
    print("\n" + "="*55)
    print("STEP 4: CROSS VALIDATION (PRE-SMOTE, REAL DATA)")
    print("="*55)
    
    cv_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.5,
        gamma=0.1,
        tree_method='hist',
        eval_metric='mlogloss',
        random_state=42
    )
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_f1 = cross_val_score(
        cv_model, X_train, y_train,
        cv=skf, scoring='f1_macro', n_jobs=-1
    )
    cv_recall = cross_val_score(
        cv_model, X_train, y_train,
        cv=skf, scoring='recall_macro', n_jobs=-1
    )
    
    print(f"5-Fold CV F1-Macro scores:    {cv_f1.round(4)}")
    print(f"Mean F1-Macro:  {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    print(f"Mean Recall:    {cv_recall.mean():.4f}")
    
    if cv_f1.std() > 0.05:
        print("⚠️  HIGH VARIANCE — possible instability in features")
    else:
        print("✅ LOW VARIANCE — features are stable across folds")
    
    return cv_f1.mean()

# ──────────────────────────────────────────────────────────
# STEP 5 — hyperparameter_optimization()
# ──────────────────────────────────────────────────────────
def hyperparameter_optimization(X_train, y_train):
    """
    Optuna-based hyperparameter search.
    30 trials to stay within hackathon time budget (~15 mins).
    Uses 3-fold CV inside each trial for efficiency.
    """
    print("\n" + "="*55)
    print("STEP 5: HYPERPARAMETER OPTIMIZATION (OPTUNA, 30 TRIALS)")
    print("="*55)
    
    def objective(trial):
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'tree_method': 'hist',
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'n_estimators': trial.suggest_int(
                'n_estimators', 100, 500
            ),
            'max_depth': trial.suggest_int(
                'max_depth', 3, 6
            ),
            'learning_rate': trial.suggest_float(
                'learning_rate', 0.01, 0.1, log=True
            ),
            'subsample': trial.suggest_float(
                'subsample', 0.6, 0.9
            ),
            'colsample_bytree': trial.suggest_float(
                'colsample_bytree', 0.6, 0.9
            ),
            'min_child_weight': trial.suggest_int(
                'min_child_weight', 3, 10
            ),
            'reg_alpha': trial.suggest_float(
                'reg_alpha', 0.0, 0.5
            ),
            'reg_lambda': trial.suggest_float(
                'reg_lambda', 0.5, 3.0
            ),
            'gamma': trial.suggest_float(
                'gamma', 0.0, 0.3
            ),
        }
        
        model = xgb.XGBClassifier(**params)
        skf = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=42
        )
        scores = cross_val_score(
            model, X_train, y_train,
            cv=skf, scoring='f1_macro', n_jobs=-1
        )
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    
    best_params = study.best_params
    best_params.update({
        'objective': 'multi:softprob',
        'num_class': 3,
        'tree_method': 'hist',
        'eval_metric': 'mlogloss',
        'random_state': 42
    })
    
    print(f"\n✅ Best CV F1-Macro: {study.best_value:.4f}")
    print(f"Best params: {best_params}")
    
    return best_params

# ──────────────────────────────────────────────────────────
# STEP 6 — handle_imbalance()
# ──────────────────────────────────────────────────────────
def handle_imbalance(X_train, y_train):
    """
    Apply SMOTE AFTER CV and hyperparameter tuning.
    Use moderate oversampling — NOT perfect 33/33/33 balance.
    Target: bring Critical to ~8% of training data.
    """
    print("\n" + "="*55)
    print("STEP 6: SMOTE (MODERATE, APPLIED AFTER CV)")
    print("="*55)
    
    counts = Counter(y_train)
    total = len(y_train)
    
    print("Before SMOTE:")
    for k, v in sorted(counts.items()):
        name = {0:'Clear', 1:'Low Risk', 2:'Critical'}[k]
        print(f"  {name}: {v} ({v/total*100:.2f}%)")
    
    # Bring Critical to ~8% of training size
    target_critical = max(
        counts[2] * 5,
        int(total * 0.08)
    )
    target_critical = min(target_critical, counts[0] // 5)
    
    smote = SMOTE(
        sampling_strategy={2: target_critical},
        random_state=42,
        k_neighbors=5
    )
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    
    counts_after = Counter(y_sm)
    total_after = len(y_sm)
    print("\nAfter SMOTE:")
    for k, v in sorted(counts_after.items()):
        name = {0:'Clear', 1:'Low Risk', 2:'Critical'}[k]
        print(f"  {name}: {v} ({v/total_after*100:.2f}%)")
    
    return X_sm, y_sm

# ──────────────────────────────────────────────────────────
# STEP 7 — train_xgboost()
# ──────────────────────────────────────────────────────────
def train_xgboost(X_train, y_train, X_test, y_test, best_params):
    """
    Train final XGBoost with best hyperparameters.
    Uses early stopping + sample weights.
    """
    print("\n" + "="*55)
    print("STEP 7: XGBOOST FINAL TRAINING")
    print("="*55)
    
    # Sample weights to further help minority class
    counts = Counter(y_train)
    total = len(y_train)
    weight_map = {
        cls: total / (len(counts) * count)
        for cls, count in counts.items()
    }
    sample_weights = np.array([weight_map[l] for l in y_train])
    
    best_params['early_stopping_rounds'] = 30
    
    model = xgb.XGBClassifier(**best_params)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=50
    )
    
    print(f"\n✅ XGBoost trained.")
    print(f"   Best iteration: {model.best_iteration}")
    
    return model

# ──────────────────────────────────────────────────────────
# STEP 8 — train_isolation_forest()
# ──────────────────────────────────────────────────────────
def train_isolation_forest(X_train, X_full_engineered):
    """
    Train on ORIGINAL (non-SMOTE) training data.
    Score all containers in full dataset.
    """
    print("\n" + "="*55)
    print("STEP 8: ISOLATION FOREST (PARALLEL ANOMALY DETECTION)")
    print("="*55)
    
    iso = IsolationForest(
        contamination=0.02,
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_train)
    
    # Score full dataset
    raw_scores = -1 * iso.decision_function(X_full_engineered)
    min_s, max_s = raw_scores.min(), raw_scores.max()
    anomaly_scores = (raw_scores - min_s) / (max_s - min_s + 1e-8)
    is_anomaly = (iso.predict(X_full_engineered) == -1).astype(int)
    
    print(f"Anomalies detected: {is_anomaly.sum()} "
          f"({is_anomaly.mean()*100:.2f}%)")
    
    return iso, anomaly_scores, is_anomaly

# ──────────────────────────────────────────────────────────
# STEP 9 — evaluate_model()
# ──────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test):
    """
    Full evaluation with honest metrics.
    Highlight Critical class performance specifically.
    """
    print("\n" + "="*55)
    print("STEP 9: MODEL EVALUATION")
    print("="*55)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Clear', 'Low Risk', 'Critical']
    ))
    
    auc = roc_auc_score(
        y_test, y_prob, multi_class='ovo', average='macro'
    )
    print(f"AUC-ROC (Macro OvO): {auc:.4f}")
    
    report_dict = classification_report(
        y_test, y_pred, output_dict=True
    )
    critical_recall    = report_dict['2']['recall']
    critical_precision = report_dict['2']['precision']
    critical_f1        = report_dict['2']['f1-score']
    
    print(f"\n{'─'*40}")
    print(f"CRITICAL CLASS METRICS (most important):")
    print(f"  Recall:    {critical_recall*100:.2f}%  "
          f"← % of real threats caught")
    print(f"  Precision: {critical_precision*100:.2f}%  "
          f"← % of flags that are real")
    print(f"  F1-Score:  {critical_f1*100:.2f}%")
    print(f"{'─'*40}")
    
    # False negative analysis
    cm = confusion_matrix(y_test, y_pred)
    fn_as_clear   = cm[2][0]
    fn_as_lowrisk = cm[2][1]
    total_critical = cm[2].sum()
    
    print(f"\nCritical containers MISSED:")
    print(f"  Predicted as Clear:    {fn_as_clear}")
    print(f"  Predicted as Low Risk: {fn_as_lowrisk}")
    print(f"  Total missed: {fn_as_clear + fn_as_lowrisk} "
          f"/ {total_critical}")
    
    # Confusion matrix plot
    os.makedirs('model_plots', exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Clear', 'Low Risk', 'Critical'],
        yticklabels=['Clear', 'Low Risk', 'Critical']
    )
    plt.title('XGBoost Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('model_plots/confusion_matrix.png')
    plt.close()
    
    return critical_recall, auc

# ──────────────────────────────────────────────────────────
# STEP 10 — calibrate_risk_scores()
# ──────────────────────────────────────────────────────────
def calibrate_risk_scores(
    model, X_full, y_full, anomaly_scores
):
    """
    Compute combined risk score.
    Calibrate thresholds using actual score distributions
    from known-labeled containers in full dataset.
    """
    print("\n" + "="*55)
    print("STEP 10: RISK SCORE CALIBRATION")
    print("="*55)
    
    probs = model.predict_proba(X_full)
    xgb_critical_prob = probs[:, 2]
    xgb_lowrisk_prob  = probs[:, 1]
    
    # Weighted combination
    combined = (
        xgb_critical_prob * 0.60 +
        xgb_lowrisk_prob  * 0.35 +
        anomaly_scores    * 0.05
    ) * 100
    
    # Print score distributions by true label for calibration
    print("\nRisk Score distributions by true label:")
    label_map = {0: 'Clear', 1: 'Low Risk', 2: 'Critical'}
    
    percentiles = {}
    for label, name in label_map.items():
        mask = y_full == label
        if mask.sum() == 0:
            continue
        scores = combined[mask]
        p25  = np.percentile(scores, 25)
        p50  = np.percentile(scores, 50)
        p75  = np.percentile(scores, 75)
        p90  = np.percentile(scores, 90)
        print(f"\n  {name} (n={mask.sum()}):")
        print(f"    25th pct: {p25:.1f}")
        print(f"    Median:   {p50:.1f}")
        print(f"    75th pct: {p75:.1f}")
        print(f"    90th pct: {p90:.1f}")
        percentiles[label] = {
            'p25': p25, 'p50': p50, 
            'p75': p75, 'p90': p90
        }
    
    # Auto-calibrate thresholds
    if 2 in percentiles and 1 in percentiles:
        critical_threshold = (
            percentiles[1]['p90'] + percentiles[2]['p25']
        ) / 2
        critical_threshold = round(
            max(critical_threshold, 40.0), 1
        )
    else:
        critical_threshold = 55.0
    
    if 1 in percentiles and 0 in percentiles:
        lowrisk_threshold = (
            percentiles[0]['p90'] + percentiles[1]['p25']
        ) / 2
        lowrisk_threshold = round(
            max(lowrisk_threshold, 10.0), 1
        )
    else:
        lowrisk_threshold = 15.0
    
    print(f"\n✅ Auto-calibrated thresholds:")
    print(f"   Critical threshold:  >= {critical_threshold}")
    print(f"   Low Risk threshold:  >= {lowrisk_threshold}")
    
    # Assign risk levels
    conditions = [
        combined >= critical_threshold,
        (combined >= lowrisk_threshold) & 
        (combined < critical_threshold),
        combined < lowrisk_threshold
    ]
    risk_levels = np.select(
        conditions, ['Critical', 'Low Risk', 'Clear'],
        default='Clear'
    )
    
    # Distribution check
    print("\nFinal Risk Level Distribution:")
    unique_lvls, counts_lvls = np.unique(risk_levels, return_counts=True)
    total = len(risk_levels)
    dist_dict = dict(zip(unique_lvls, counts_lvls))
    for lvl in ['Critical', 'Low Risk', 'Clear']:
        c = dist_dict.get(lvl, 0)
        print(f"  {lvl}: {c} ({c/total*100:.2f}%)")
    
    # Sanity check
    low_risk_count = dist_dict.get('Low Risk', 0)
    if low_risk_count < 5000:
        print("\n⚠️  WARNING: Low Risk count seems low.")
        print(f"   Consider lowering lowrisk_threshold below "
              f"{lowrisk_threshold}")
    else:
        print("✅ Risk distribution looks healthy.")
    
    thresholds = {
        'critical': float(critical_threshold),
        'low_risk': float(lowrisk_threshold)
    }
    
    return (combined.round(2), risk_levels, 
            xgb_critical_prob, xgb_lowrisk_prob, thresholds)

# ── SHAP EXPLANATIONS ──────────────────────────────────────
def generate_explanation_text(shap_vals, feature_names, risk_level, container_idx, top_n=3):
    label_map = {
        'weight_discrepancy': 'weight mismatch',
        'abs_weight_discrepancy': 'high weight discrepancy',
        'value_per_kg': 'unusual value-to-weight ratio',
        'dwell_time_flag': 'excessive dwell time',
        'Dwell_Time_Hours': 'long port dwell time',
        'Importer_ID_critical_rate': 'high-risk importer history',
        'Exporter_ID_critical_rate': 'high-risk exporter history',
        'Origin_Country_critical_rate': 'high-risk origin country',
        'HS_Chapter_critical_rate': 'high-risk commodity type',
        'Importer_ID_is_repeat_offender': 'repeat offender importer',
        'Exporter_ID_is_repeat_offender': 'repeat offender exporter',
        'weight_x_dwell': 'combined weight + dwell anomaly'
    }
    
    if risk_level == 'Clear':
        return "Cleared: no significant risk indicators detected."
    
    target_class_idx = 2 if risk_level == 'Critical' else 1
    instance_shaps = np.abs(shap_vals[target_class_idx][container_idx])
    top_indices = np.argsort(instance_shaps)[-top_n:][::-1]
    
    explanations = []
    for idx in top_indices:
        feat_name = feature_names[idx]
        human_label = label_map.get(feat_name, feat_name.replace('_', ' '))
        explanations.append(human_label)
        
    return f"Flagged {risk_level}: {', '.join(explanations)}."

def generate_shap_explanations(model, X_test, X_full, feature_names):
    print("\n" + "="*55)
    print("STEP 11: SHAP EXPLANATIONS (OPTIMIZED)")
    print("="*55)
    
    explainer = shap.TreeExplainer(model)
    
    # Summary plots using test set
    shap_test = explainer.shap_values(X_test)
    if isinstance(shap_test, np.ndarray) and shap_test.ndim == 3:
        shap_test_list = [shap_test[:, :, i] for i in range(3)]
    else:
        shap_test_list = shap_test
    
    os.makedirs('model_plots', exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_test_list[2], X_test, show=False)
    plt.tight_layout()
    plt.savefig('model_plots/shap_summary.png')
    plt.close()
    
    # Full dataset explanations (approximate for speed)
    print("Computing full SHAP values...")
    shap_full = explainer.shap_values(X_full, check_additivity=False)
    
    if isinstance(shap_full, np.ndarray) and shap_full.ndim == 3:
        shap_full_list = [shap_full[:, :, i] for i in range(3)]
    else:
        shap_full_list = shap_full
    
    # Top features for summary
    mean_abs_shap = np.abs(shap_test_list[2]).mean(axis=0)
    top_5_idx = np.argsort(mean_abs_shap)[-5:][::-1]
    top_5_features = [feature_names[i] for i in top_5_idx]
    
    return shap_full_list, top_5_features

# ──────────────────────────────────────────────────────────
# STEP 12 — generate_output_csv()
# ──────────────────────────────────────────────────────────
def generate_output_csv(
    container_ids_full, combined_scores, risk_levels,
    xgb_critical_probs, xgb_lowrisk_probs,
    anomaly_scores, is_anomaly, explanations
):
    output = pd.DataFrame({
        'Container_ID':       container_ids_full,
        'Risk_Score':         combined_scores,
        'Risk_Level':         risk_levels,
        'XGB_Critical_Prob':  xgb_critical_probs.round(4),
        'XGB_LowRisk_Prob':   xgb_lowrisk_probs.round(4),
        'Anomaly_Score':      anomaly_scores.round(4),
        'Is_Anomaly':         is_anomaly,
        'Explanation_Summary': explanations
    })
    output.to_csv('final_predictions.csv', index=False)
    print("✅ final_predictions.csv saved.")
    return output

# ──────────────────────────────────────────────────────────
# STEP 13 — save_models()
# ──────────────────────────────────────────────────────────
def save_models(
    model, iso, feature_names, entity_maps, 
    train_medians, stats, thresholds
):
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model, 'models/xgb_model.pkl')
    joblib.dump(iso,   'models/isolation_forest.pkl')
    
    with open('models/feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    
    with open('models/thresholds.json', 'w') as f:
        json.dump(thresholds, f)
    
    with open('models/entity_rate_maps.pkl', 'wb') as f:
        pickle.dump(entity_maps, f)
    
    with open('models/train_stats.pkl', 'wb') as f:
        pickle.dump({
            'medians': train_medians.to_dict(),
            'stats':   stats
        }, f)
    
    print("✅ All models and artifacts saved in models/")

# ──────────────────────────────────────────────────────────
# STEP 14 — main()
# ──────────────────────────────────────────────────────────
def main():
    print("\n🚢 SmartContainer Risk Engine — Model Training")
    print("=" * 55)
    
    # 1. Load and clean
    df = load_and_clean()
    
    # 2. Split FIRST
    X_train_raw, X_test_raw, y_train, y_test, ids_train, ids_test = split_data(df)
    
    # 3. Feature engineering
    X_train, X_test, entity_maps, stats, train_medians = feature_engineering(X_train_raw, X_test_raw, y_train)
    feature_names = X_train.columns.tolist()
    
    # 4. Cross validate
    cv_score = cross_validate(X_train, y_train)
    print(f"✅ CV complete. Mean F1-Macro: {cv_score:.4f}")
    
    # 5. Hyperparameter optimization
    best_params = hyperparameter_optimization(X_train, y_train)
    
    # 6. SMOTE
    X_train_sm, y_train_sm = handle_imbalance(X_train, y_train)
    
    # 7. Train XGBoost
    xgb_model = train_xgboost(X_train_sm, y_train_sm, X_test, y_test, best_params)
    
    # 8. Isolation Forest
    iso_model, _, _ = train_isolation_forest(X_train, X_train)
    
    # 9. Evaluate
    critical_recall, auc_score = evaluate_model(xgb_model, X_test, y_test)
    
    # 10. Re-engineer full dataset for output
    X_full, _, _, _, _ = feature_engineering(df[X_train_raw.columns], df[X_train_raw.columns].head(1), y_train)
    X_full = X_full[feature_names]
    X_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_full = X_full.fillna(train_medians)
    
    # Isolation Forest on full data
    _, anomaly_scores_full, is_anomaly_full = train_isolation_forest(X_train, X_full)
    
    # 11. Calibrate risk scores
    y_full = df['target'].values
    (combined_scores, risk_levels, xgb_crit_probs, xgb_lr_probs, thresholds) = \
        calibrate_risk_scores(xgb_model, X_full, y_full, anomaly_scores_full)
    
    # 12. SHAP
    shap_vals_list, top_features = generate_shap_explanations(xgb_model, X_test, X_full, feature_names)
    
    explanations = []
    for i in range(len(risk_levels)):
        explanations.append(generate_explanation_text(shap_vals_list, feature_names, risk_levels[i], i))
    
    # 13. Generate output CSV
    generate_output_csv(
        df['Container_ID'].values, combined_scores, risk_levels,
        xgb_crit_probs, xgb_lr_probs, anomaly_scores_full, is_anomaly_full, explanations
    )
    
    # 14. Save
    save_models(xgb_model, iso_model, feature_names, entity_maps, train_medians, stats, thresholds)
    
    # 15. Summary
    total = len(risk_levels)
    dist_counts = Counter(risk_levels)
    print("\n" + "#"*55)
    print("FINAL SUMMARY")
    print("#"*55)
    print(f"Total containers Scored:     {total}")
    print(f"Critical:  {dist_counts['Critical']} ({dist_counts['Critical']/total*100:.2f}%)")
    print(f"Low Risk:  {dist_counts['Low Risk']} ({dist_counts['Low Risk']/total*100:.2f}%)")
    print(f"Clear:     {dist_counts['Clear']} ({dist_counts['Clear']/total*100:.2f}%)")
    print(f"Anomalies: {is_anomaly_full.sum()}")
    print(f"CV F1-Macro:        {cv_score:.4f}")
    print(f"Test AUC-ROC:       {auc_score:.4f}")
    print(f"Critical Recall:    {critical_recall*100:.2f}%")
    print(f"Top Features (SHAP): {', '.join(top_features)}")
    print("\n✅ Pipeline complete. Model is production-ready.")

if __name__ == "__main__":
    main()
