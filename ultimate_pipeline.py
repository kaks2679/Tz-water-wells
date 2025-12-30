#!/usr/bin/env python3
"""
Ultimate Ensemble Model - Target: 0.83+
Combines multiple LightGBM models with different configurations
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ULTIMATE ENSEMBLE MODEL - TARGET: 0.83+")
print("=" * 70)

# Load data
print("\n[1/6] Loading data...")
train_values = pd.read_csv('4910797b-ee55-40a7-8668-10efd5c1b960.csv')
train_labels = pd.read_csv('0bf8bc6e-30d0-4c50-956a-603fc693d966.csv')
test_values = pd.read_csv('702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv')

train_df = train_values.merge(train_labels, on='id')
print(f"Training: {len(train_df):,} | Test: {len(test_values):,}")

# Feature engineering (same as enhanced)
print("\n[2/6] Feature engineering...")

def engineer_features_v2(df):
    df = df.copy()
    
    df['date_recorded'] = pd.to_datetime(df['date_recorded'])
    df['year_recorded'] = df['date_recorded'].dt.year
    df['month_recorded'] = df['date_recorded'].dt.month
    df['day_of_year'] = df['date_recorded'].dt.dayofyear
    df['age'] = df['year_recorded'] - df['construction_year']
    df['age_squared'] = df['age'] ** 2
    
    df['gps_height_zero'] = (df['gps_height'] == 0).astype(int)
    df['location_missing'] = ((df['latitude'] == 0) | (df['longitude'] == 0)).astype(int)
    df['lat_lon_ratio'] = df['latitude'] / (df['longitude'].abs() + 1e-6)
    df['gps_lat_interaction'] = df['gps_height'] * df['latitude']
    df['gps_lon_interaction'] = df['gps_height'] * df['longitude']
    
    center_lat, center_lon = df['latitude'].median(), df['longitude'].median()
    df['distance_from_center'] = np.sqrt(
        (df['latitude'] - center_lat)**2 + (df['longitude'] - center_lon)**2
    )
    
    df['log_population'] = np.log1p(df['population'])
    df['population_zero'] = (df['population'] == 0).astype(int)
    df['population_density'] = df['population'] / (df['gps_height'].abs() + 1)
    
    df['log_amount'] = np.log1p(df['amount_tsh'])
    df['amount_zero'] = (df['amount_tsh'] == 0).astype(int)
    df['amount_per_person'] = df['amount_tsh'] / (df['population'] + 1)
    
    df['extraction_payment'] = df['extraction_type_class'] + '_' + df['payment_type']
    df['source_quality'] = df['source_type'] + '_' + df['water_quality']
    df['region_basin'] = df['region'] + '_' + df['basin']
    df['quantity_quality'] = df['quantity'] + '_' + df['quality_group']
    df['waterpoint_extraction'] = df['waterpoint_type'] + '_' + df['extraction_type_class']
    df['management_payment'] = df['management'] + '_' + df['payment_type']
    
    df['construction_decade'] = (df['construction_year'] // 10) * 10
    df['is_new_well'] = (df['construction_year'] >= 2005).astype(int)
    df['is_old_well'] = (df['construction_year'] <= 1980).astype(int)
    
    df = df.drop('date_recorded', axis=1)
    return df

train_df = engineer_features_v2(train_df)
test_values = engineer_features_v2(test_values)

important_features = [
    'amount_tsh', 'gps_height', 'longitude', 'latitude', 'population',
    'year_recorded', 'month_recorded', 'construction_year',
    'age', 'age_squared', 'day_of_year',
    'log_population', 'log_amount', 'population_density', 'amount_per_person',
    'distance_from_center', 'lat_lon_ratio', 
    'gps_lat_interaction', 'gps_lon_interaction',
    'gps_height_zero', 'location_missing', 'population_zero', 'amount_zero',
    'is_new_well', 'is_old_well',
    'quantity', 'quality_group', 'waterpoint_type', 'source_type',
    'extraction_type_class', 'payment_type', 'water_quality',
    'basin', 'region', 'scheme_management', 'extraction_type',
    'management', 'source_class', 'waterpoint_type_group',
    'extraction_payment', 'source_quality', 'region_basin',
    'quantity_quality', 'waterpoint_extraction', 'management_payment',
    'construction_decade'
]

# Preprocessing
print("\n[3/6] Preprocessing...")

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

label_encoders = {}
cat_features = train_df[important_features].select_dtypes(include=['object']).columns

for col in cat_features:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    test_col = test_values[col].astype(str)
    test_values[col] = test_col.map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    label_encoders[col] = le

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(train_df['status_group'])

X = train_df[important_features].values
X_test = test_values[important_features].values

print(f"Feature matrix: {X.shape}")

# Multiple model configurations
print("\n[4/6] Training ensemble of 3 models...")

model_configs = [
    {
        'name': 'Deep',
        'params': {
            'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
            'boosting_type': 'gbdt', 'num_leaves': 60, 'learning_rate': 0.02,
            'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
            'max_depth': 25, 'min_child_samples': 10, 'lambda_l1': 0.05, 'lambda_l2': 0.05,
            'verbose': -1, 'n_jobs': -1
        }
    },
    {
        'name': 'Balanced',
        'params': {
            'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
            'boosting_type': 'gbdt', 'num_leaves': 50, 'learning_rate': 0.03,
            'feature_fraction': 0.85, 'bagging_fraction': 0.85, 'bagging_freq': 5,
            'max_depth': 20, 'min_child_samples': 15, 'lambda_l1': 0.05, 'lambda_l2': 0.05,
            'verbose': -1, 'n_jobs': -1
        }
    },
    {
        'name': 'Conservative',
        'params': {
            'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
            'boosting_type': 'gbdt', 'num_leaves': 40, 'learning_rate': 0.05,
            'feature_fraction': 0.9, 'bagging_fraction': 0.9, 'bagging_freq': 5,
            'max_depth': 15, 'min_child_samples': 20, 'lambda_l1': 0.1, 'lambda_l2': 0.1,
            'verbose': -1, 'n_jobs': -1
        }
    }
]

all_test_preds = []
cv_scores_all = []

for config in model_configs:
    print(f"\nModel: {config['name']}")
    print("-" * 60)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    fold_models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
        
        model = lgb.train(
            config['params'],
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )
        
        y_pred = model.predict(X_val_fold)
        y_pred_class = y_pred.argmax(axis=1)
        acc = accuracy_score(y_val_fold, y_pred_class)
        cv_scores.append(acc)
        fold_models.append(model)
        print(f"  Fold {fold}: {acc:.4f} (iter: {model.best_iteration})")
    
    mean_cv = np.mean(cv_scores)
    cv_scores_all.append(mean_cv)
    print(f"  Mean CV: {mean_cv:.4f} (±{np.std(cv_scores):.4f})")
    
    # Train final model and predict
    final_train_data = lgb.Dataset(X, label=y)
    final_model = lgb.train(
        config['params'],
        final_train_data,
        num_boost_round=int(np.mean([m.best_iteration for m in fold_models]))
    )
    
    test_pred = final_model.predict(X_test)
    all_test_preds.append(test_pred)

# Ensemble predictions
print("\n[5/6] Creating ensemble...")

# Weighted average (weight by CV score)
weights = np.array(cv_scores_all) / np.sum(cv_scores_all)
ensemble_pred = np.zeros_like(all_test_preds[0])

for i, pred in enumerate(all_test_preds):
    ensemble_pred += weights[i] * pred
    print(f"Model {model_configs[i]['name']}: weight = {weights[i]:.4f}")

ensemble_pred_class = ensemble_pred.argmax(axis=1)
test_predictions_labels = target_encoder.inverse_transform(ensemble_pred_class)

# Save submission
print("\n[6/6] Saving submission...")

submission = pd.DataFrame({
    'id': test_values['id'],
    'status_group': test_predictions_labels
})

submission.to_csv('ultimate_submission.csv', index=False)

print(f"\n✅ Ultimate submission saved: ultimate_submission.csv")
print(f"Predictions: {submission.shape[0]:,}")
print(f"\nDistribution:")
print(submission['status_group'].value_counts(normalize=True))

print("\n" + "=" * 70)
print("ULTIMATE ENSEMBLE COMPLETE!")
print("=" * 70)
print(f"Ensemble CV (weighted): {np.average(cv_scores_all, weights=weights):.4f}")
print(f"Individual CVs: {[f'{s:.4f}' for s in cv_scores_all]}")
print(f"\nPrevious: 0.8070 → Enhanced: 0.8088 expected")
print(f"Target: 0.8299+ (leaderboard #1)")
print(f"\n🎯 Submit: ultimate_submission.csv")
