#!/usr/bin/env python3
"""
Tanzanian Water Wells - LightGBM Pipeline
Fast, efficient gradient boosting approach
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TANZANIAN WATER WELLS - LIGHTGBM PIPELINE")
print("=" * 60)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")
train_values = pd.read_csv('4910797b-ee55-40a7-8668-10efd5c1b960.csv')
train_labels = pd.read_csv('0bf8bc6e-30d0-4c50-956a-603fc693d966.csv')
test_values = pd.read_csv('702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv')

train_df = train_values.merge(train_labels, on='id')
print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_values)}")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
print("\n[2/6] Engineering features...")

def engineer_features(df):
    df = df.copy()
    
    # Date features
    df['date_recorded'] = pd.to_datetime(df['date_recorded'])
    df['year_recorded'] = df['date_recorded'].dt.year
    df['month_recorded'] = df['date_recorded'].dt.month
    df['age'] = df['year_recorded'] - df['construction_year']
    
    # Geographic
    df['gps_height_zero'] = (df['gps_height'] == 0).astype(int)
    df['location_missing'] = ((df['latitude'] == 0) | (df['longitude'] == 0)).astype(int)
    
    # Combinations
    df['extraction_payment'] = df['extraction_type_class'] + '_' + df['payment_type']
    df['source_quality'] = df['source_type'] + '_' + df['water_quality']
    df['region_basin'] = df['region'] + '_' + df['basin']
    
    # Population
    df['log_population'] = np.log1p(df['population'])
    df['population_zero'] = (df['population'] == 0).astype(int)
    
    df = df.drop('date_recorded', axis=1)
    return df

train_df = engineer_features(train_df)
test_values = engineer_features(test_values)

# ============================================================================
# 3. PREPROCESSING
# ============================================================================
print("\n[3/6] Preprocessing...")

important_features = [
    'amount_tsh', 'gps_height', 'longitude', 'latitude', 'population',
    'year_recorded', 'month_recorded', 'age', 'log_population',
    'gps_height_zero', 'location_missing', 'population_zero',
    'quantity', 'quality_group', 'waterpoint_type', 'source_type',
    'extraction_type_class', 'payment_type', 'water_quality',
    'basin', 'region', 'scheme_management', 'extraction_type',
    'management', 'extraction_payment', 'source_quality',
    'source_class', 'waterpoint_type_group', 'region_basin'
]

# Fill missing
def fill_missing(df):
    df = df.copy()
    numeric_cols = df[important_features].select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    cat_cols = df[important_features].select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col].fillna('unknown', inplace=True)
    
    return df

train_df = fill_missing(train_df)
test_values = fill_missing(test_values)

# Encode categoricals
label_encoders = {}
cat_features = train_df[important_features].select_dtypes(include=['object']).columns

for col in cat_features:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    
    test_col = test_values[col].astype(str)
    test_values[col] = test_col.map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    
    label_encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(train_df['status_group'])

X = train_df[important_features].values
X_test = test_values[important_features].values

print(f"Feature matrix: {X.shape}")

# ============================================================================
# 4. LIGHTGBM TRAINING
# ============================================================================
print("\n[4/6] Training LightGBM with 5-Fold CV...")

lgb_params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 40,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 15,
    'min_child_samples': 20,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1,
    'n_jobs': -1
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
    
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    y_pred = model.predict(X_val_fold)
    y_pred_class = y_pred.argmax(axis=1)
    acc = accuracy_score(y_val_fold, y_pred_class)
    cv_scores.append(acc)
    models.append(model)
    print(f"  Fold {fold}: {acc:.4f} (iterations: {model.best_iteration})")

print(f"\nMean CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# ============================================================================
# 5. FINAL MODEL
# ============================================================================
print("\n[5/6] Training final model...")

final_train_data = lgb.Dataset(X, label=y)
final_model = lgb.train(
    lgb_params,
    final_train_data,
    num_boost_round=int(np.mean([m.best_iteration for m in models]))
)

print("Final model trained!")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': important_features,
    'importance': final_model.feature_importance()
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# 6. PREDICTIONS
# ============================================================================
print("\n[6/6] Generating predictions...")

test_pred = final_model.predict(X_test)
test_pred_class = test_pred.argmax(axis=1)
test_predictions_labels = target_encoder.inverse_transform(test_pred_class)

submission = pd.DataFrame({
    'id': test_values['id'],
    'status_group': test_predictions_labels
})

submission.to_csv('lightgbm_submission.csv', index=False)
print(f"\nSubmission file saved: lightgbm_submission.csv")
print(f"Prediction distribution:\n{submission['status_group'].value_counts(normalize=True)}")

print("\n" + "=" * 60)
print("LIGHTGBM PIPELINE COMPLETE!")
print("=" * 60)
print(f"\nCV Accuracy: {np.mean(cv_scores):.4f}")
print(f"Expected improvement over baseline (0.7573): +{(np.mean(cv_scores) - 0.7573)*100:.2f}%")
print("\nFiles created:")
print("  - lightgbm_submission.csv (main submission)")
print("  - improved_submission.csv (Random Forest)")
