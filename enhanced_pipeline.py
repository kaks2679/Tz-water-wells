#!/usr/bin/env python3
"""
Enhanced LightGBM Pipeline - Target: 0.83+
Additional optimizations to reach top leaderboard
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
print("ENHANCED LIGHTGBM PIPELINE - TARGET: 0.83+")
print("=" * 70)

# Load data
print("\n[1/7] Loading data...")
train_values = pd.read_csv('4910797b-ee55-40a7-8668-10efd5c1b960.csv')
train_labels = pd.read_csv('0bf8bc6e-30d0-4c50-956a-603fc693d966.csv')
test_values = pd.read_csv('702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv')

train_df = train_values.merge(train_labels, on='id')
print(f"Training: {len(train_df):,} | Test: {len(test_values):,}")

# Enhanced feature engineering
print("\n[2/7] Enhanced feature engineering...")

def engineer_features_v2(df):
    """Enhanced feature engineering with additional interactions"""
    df = df.copy()
    
    # Date features
    df['date_recorded'] = pd.to_datetime(df['date_recorded'])
    df['year_recorded'] = df['date_recorded'].dt.year
    df['month_recorded'] = df['date_recorded'].dt.month
    df['day_of_year'] = df['date_recorded'].dt.dayofyear
    df['age'] = df['year_recorded'] - df['construction_year']
    df['age_squared'] = df['age'] ** 2
    
    # Geographic features
    df['gps_height_zero'] = (df['gps_height'] == 0).astype(int)
    df['location_missing'] = ((df['latitude'] == 0) | (df['longitude'] == 0)).astype(int)
    df['lat_lon_ratio'] = df['latitude'] / (df['longitude'].abs() + 1e-6)
    df['gps_lat_interaction'] = df['gps_height'] * df['latitude']
    df['gps_lon_interaction'] = df['gps_height'] * df['longitude']
    
    # Distance from center (rough estimate)
    center_lat, center_lon = df['latitude'].median(), df['longitude'].median()
    df['distance_from_center'] = np.sqrt(
        (df['latitude'] - center_lat)**2 + (df['longitude'] - center_lon)**2
    )
    
    # Population features
    df['log_population'] = np.log1p(df['population'])
    df['population_zero'] = (df['population'] == 0).astype(int)
    df['population_density'] = df['population'] / (df['gps_height'].abs() + 1)
    
    # Amount features
    df['log_amount'] = np.log1p(df['amount_tsh'])
    df['amount_zero'] = (df['amount_tsh'] == 0).astype(int)
    df['amount_per_person'] = df['amount_tsh'] / (df['population'] + 1)
    
    # Categorical combinations
    df['extraction_payment'] = df['extraction_type_class'] + '_' + df['payment_type']
    df['source_quality'] = df['source_type'] + '_' + df['water_quality']
    df['region_basin'] = df['region'] + '_' + df['basin']
    df['quantity_quality'] = df['quantity'] + '_' + df['quality_group']
    df['waterpoint_extraction'] = df['waterpoint_type'] + '_' + df['extraction_type_class']
    df['management_payment'] = df['management'] + '_' + df['payment_type']
    
    # Construction year bins
    df['construction_decade'] = (df['construction_year'] // 10) * 10
    df['is_new_well'] = (df['construction_year'] >= 2005).astype(int)
    df['is_old_well'] = (df['construction_year'] <= 1980).astype(int)
    
    df = df.drop('date_recorded', axis=1)
    return df

train_df = engineer_features_v2(train_df)
test_values = engineer_features_v2(test_values)

print(f"Enhanced features created!")

# Feature selection
print("\n[3/7] Selecting features...")

important_features = [
    # Original numeric
    'amount_tsh', 'gps_height', 'longitude', 'latitude', 'population',
    'year_recorded', 'month_recorded', 'construction_year',
    
    # Enhanced numeric
    'age', 'age_squared', 'day_of_year',
    'log_population', 'log_amount', 'population_density', 'amount_per_person',
    'distance_from_center', 'lat_lon_ratio', 
    'gps_lat_interaction', 'gps_lon_interaction',
    
    # Flags
    'gps_height_zero', 'location_missing', 'population_zero', 'amount_zero',
    'is_new_well', 'is_old_well',
    
    # Original categorical
    'quantity', 'quality_group', 'waterpoint_type', 'source_type',
    'extraction_type_class', 'payment_type', 'water_quality',
    'basin', 'region', 'scheme_management', 'extraction_type',
    'management', 'source_class', 'waterpoint_type_group',
    
    # Enhanced categorical combinations
    'extraction_payment', 'source_quality', 'region_basin',
    'quantity_quality', 'waterpoint_extraction', 'management_payment',
    'construction_decade'
]

print(f"Total features: {len(important_features)}")

# Preprocessing
print("\n[4/7] Preprocessing...")

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

# Encoding
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

# Enhanced LightGBM parameters
print("\n[5/7] Training enhanced LightGBM...")

lgb_params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 50,  # Increased
    'learning_rate': 0.03,  # Decreased for better learning
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'max_depth': 20,  # Increased
    'min_child_samples': 15,  # Decreased
    'lambda_l1': 0.05,
    'lambda_l2': 0.05,
    'min_gain_to_split': 0.01,
    'verbose': -1,
    'n_jobs': -1
}

# Cross-validation
skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)  # More folds
cv_scores = []
models = []

print("Cross-Validation (7-fold):")
print("-" * 60)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
    val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
    
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1000,  # More rounds
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    
    y_pred = model.predict(X_val_fold)
    y_pred_class = y_pred.argmax(axis=1)
    acc = accuracy_score(y_val_fold, y_pred_class)
    cv_scores.append(acc)
    models.append(model)
    print(f"Fold {fold}: {acc:.4f} (iterations: {model.best_iteration})")

print("-" * 60)
print(f"Mean CV: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

# Train final model
print("\n[6/7] Training final model...")

final_train_data = lgb.Dataset(X, label=y)
final_model = lgb.train(
    lgb_params,
    final_train_data,
    num_boost_round=int(np.mean([m.best_iteration for m in models]))
)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': important_features,
    'importance': final_model.feature_importance()
}).sort_values('importance', ascending=False)

print("\nTop 10 Features:")
print(feature_importance.head(10).to_string(index=False))

# Predictions
print("\n[7/7] Generating predictions...")

test_pred = final_model.predict(X_test)
test_pred_class = test_pred.argmax(axis=1)
test_predictions_labels = target_encoder.inverse_transform(test_pred_class)

submission = pd.DataFrame({
    'id': test_values['id'],
    'status_group': test_predictions_labels
})

submission.to_csv('enhanced_submission.csv', index=False)

print(f"\n✅ Enhanced submission saved: enhanced_submission.csv")
print(f"Predictions: {submission.shape[0]:,}")
print(f"\nDistribution:")
print(submission['status_group'].value_counts(normalize=True))

print("\n" + "=" * 70)
print("ENHANCED MODEL COMPLETE!")
print("=" * 70)
print(f"CV Accuracy: {np.mean(cv_scores):.4f}")
print(f"Previous competition score: 0.8070")
print(f"Target: 0.8299+ (leaderboard #1)")
print(f"\nExpected improvement: +0.01 to +0.02")
print("Submit: enhanced_submission.csv")
