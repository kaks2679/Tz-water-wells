#!/usr/bin/env python3
"""
Tanzanian Water Wells - Improved Production Pipeline
Efficient, focused approach using proven algorithms
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TANZANIAN WATER WELLS - PRODUCTION PIPELINE")
print("=" * 60)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")
train_values = pd.read_csv('4910797b-ee55-40a7-8668-10efd5c1b960.csv')
train_labels = pd.read_csv('0bf8bc6e-30d0-4c50-956a-603fc693d966.csv')
test_values = pd.read_csv('702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv')

# Merge training data
train_df = train_values.merge(train_labels, on='id')
print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_values)}")
print(f"Target distribution:\n{train_df['status_group'].value_counts(normalize=True)}")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
print("\n[2/6] Engineering features...")

def engineer_features(df):
    """Create efficient feature set"""
    df = df.copy()
    
    # Date features
    df['date_recorded'] = pd.to_datetime(df['date_recorded'])
    df['year_recorded'] = df['date_recorded'].dt.year
    df['month_recorded'] = df['date_recorded'].dt.month
    df['age'] = df['year_recorded'] - df['construction_year']
    
    # Geographic features
    df['gps_height_zero'] = (df['gps_height'] == 0).astype(int)
    df['location_missing'] = ((df['latitude'] == 0) | (df['longitude'] == 0)).astype(int)
    
    # Categorical combinations
    df['extraction_payment'] = df['extraction_type_class'] + '_' + df['payment_type']
    df['source_quality'] = df['source_type'] + '_' + df['water_quality']
    
    # Population features
    df['log_population'] = np.log1p(df['population'])
    df['population_zero'] = (df['population'] == 0).astype(int)
    
    # Drop original date column
    df = df.drop('date_recorded', axis=1)
    
    return df

train_df = engineer_features(train_df)
test_values = engineer_features(test_values)

# ============================================================================
# 3. PREPROCESSING
# ============================================================================
print("\n[3/6] Preprocessing...")

# Select important features based on previous analysis
important_features = [
    # Numeric
    'amount_tsh', 'gps_height', 'longitude', 'latitude', 'population',
    'year_recorded', 'month_recorded', 'age', 'log_population',
    'gps_height_zero', 'location_missing', 'population_zero',
    
    # Categorical (high-cardinality handled efficiently)
    'quantity', 'quality_group', 'waterpoint_type', 'source_type',
    'extraction_type_class', 'payment_type', 'water_quality',
    'basin', 'region', 'scheme_management', 'extraction_type',
    'management', 'extraction_payment', 'source_quality',
    'source_class', 'waterpoint_type_group'
]

# Fill missing values
def fill_missing(df):
    df = df.copy()
    # Numeric: fill with median
    numeric_cols = df[important_features].select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Categorical: fill with 'unknown'
    cat_cols = df[important_features].select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col].fillna('unknown', inplace=True)
    
    return df

train_df = fill_missing(train_df)
test_values = fill_missing(test_values)

# Encode categorical variables efficiently
label_encoders = {}
cat_features = train_df[important_features].select_dtypes(include=['object']).columns

for col in cat_features:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    
    # Handle unseen categories in test
    test_col = test_values[col].astype(str)
    test_values[col] = test_col.map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    
    label_encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(train_df['status_group'])

# Prepare feature matrices
X = train_df[important_features].values
X_test = test_values[important_features].values

print(f"Feature matrix shape: {X.shape}")
print(f"Test matrix shape: {X_test.shape}")

# ============================================================================
# 4. MODEL TRAINING - RANDOM FOREST
# ============================================================================
print("\n[4/6] Training Random Forest with 5-Fold CV...")

# Configure Random Forest for speed and accuracy balance
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42,
    class_weight='balanced'
)

# 5-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    # Train
    rf_model.fit(X_train_fold, y_train_fold)
    
    # Validate
    y_pred = rf_model.predict(X_val_fold)
    acc = accuracy_score(y_val_fold, y_pred)
    cv_scores.append(acc)
    print(f"  Fold {fold}: {acc:.4f}")

print(f"\nMean CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# ============================================================================
# 5. FINAL MODEL TRAINING
# ============================================================================
print("\n[5/6] Training final model on full dataset...")

final_model = RandomForestClassifier(
    n_estimators=300,  # More trees for final model
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42,
    class_weight='balanced'
)

final_model.fit(X, y)
print("Final model trained successfully!")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': important_features,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# 6. GENERATE PREDICTIONS
# ============================================================================
print("\n[6/6] Generating predictions...")

# Predict on test set
test_predictions = final_model.predict(X_test)
test_predictions_labels = target_encoder.inverse_transform(test_predictions)

# Create submission file
submission = pd.DataFrame({
    'id': test_values['id'],
    'status_group': test_predictions_labels
})

submission.to_csv('improved_submission.csv', index=False)
print(f"\nSubmission file saved: improved_submission.csv")
print(f"Predictions shape: {submission.shape}")
print(f"Prediction distribution:\n{submission['status_group'].value_counts(normalize=True)}")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE!")
print("=" * 60)
print(f"\nExpected improvement over baseline (0.7573):")
print(f"  - CV Accuracy: {np.mean(cv_scores):.4f}")
print(f"  - Expected score: ~{np.mean(cv_scores):.4f}")
print("\nNext steps:")
print("  1. Submit 'improved_submission.csv' to competition")
print("  2. Compare with baseline score (0.7573)")
print("  3. Iterate with additional features if needed")
