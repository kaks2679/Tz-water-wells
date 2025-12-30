#!/usr/bin/env python3
"""
Convert Python scripts to Jupyter notebooks with markdown cells
"""
import nbformat as nbf

# ============================================================================
# 1. RANDOM FOREST NOTEBOOK
# ============================================================================

rf_nb = nbf.v4.new_notebook()

rf_nb.cells = [
    nbf.v4.new_markdown_cell("""# Tanzanian Water Wells - Random Forest Pipeline

## Overview
This notebook implements a Random Forest classifier to predict water well functionality in Tanzania.

**Target Performance:** 76.35% Cross-Validation Accuracy

## Key Features:
- 29 engineered features
- 5-fold stratified cross-validation
- Balanced class weighting
- Feature importance analysis"""),

    nbf.v4.new_markdown_cell("## 1. Import Libraries"),
    
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')"""),

    nbf.v4.new_markdown_cell("## 2. Load Data"),
    
    nbf.v4.new_code_cell("""# Load datasets
train_values = pd.read_csv('4910797b-ee55-40a7-8668-10efd5c1b960.csv')
train_labels = pd.read_csv('0bf8bc6e-30d0-4c50-956a-603fc693d966.csv')
test_values = pd.read_csv('702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv')

# Merge training data
train_df = train_values.merge(train_labels, on='id')

print(f"Training samples: {len(train_df):,}")
print(f"Test samples: {len(test_values):,}")
print(f"\\nTarget distribution:")
print(train_df['status_group'].value_counts(normalize=True))"""),

    nbf.v4.new_markdown_cell("""## 3. Feature Engineering

We create several types of features:
- **Date features**: Year, month, well age
- **Geographic features**: GPS flags, location validity
- **Categorical combinations**: Extraction + payment, source + quality
- **Population features**: Log transformation, zero flags"""),

    nbf.v4.new_code_cell("""def engineer_features(df):
    \"\"\"Create efficient feature set\"\"\"
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

# Apply feature engineering
train_df = engineer_features(train_df)
test_values = engineer_features(test_values)

print("Feature engineering complete!")"""),

    nbf.v4.new_markdown_cell("""## 4. Data Preprocessing

Select important features and handle missing values."""),

    nbf.v4.new_code_cell("""# Select important features
important_features = [
    # Numeric
    'amount_tsh', 'gps_height', 'longitude', 'latitude', 'population',
    'year_recorded', 'month_recorded', 'age', 'log_population',
    'gps_height_zero', 'location_missing', 'population_zero',
    
    # Categorical
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

print(f"Features selected: {len(important_features)}")"""),

    nbf.v4.new_markdown_cell("## 5. Encode Categorical Variables"),

    nbf.v4.new_code_cell("""# Encode categorical variables
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
print(f"Test matrix shape: {X_test.shape}")"""),

    nbf.v4.new_markdown_cell("""## 6. Model Training with Cross-Validation

Train Random Forest with 5-fold stratified cross-validation."""),

    nbf.v4.new_code_cell("""# Configure Random Forest
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

print("Cross-Validation Results:")
print("-" * 40)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    # Train
    rf_model.fit(X_train_fold, y_train_fold)
    
    # Validate
    y_pred = rf_model.predict(X_val_fold)
    acc = accuracy_score(y_val_fold, y_pred)
    cv_scores.append(acc)
    print(f"Fold {fold}: {acc:.4f}")

print("-" * 40)
print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")"""),

    nbf.v4.new_markdown_cell("## 7. Train Final Model"),

    nbf.v4.new_code_cell("""# Train final model on full dataset
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
print("Final model trained successfully!")"""),

    nbf.v4.new_markdown_cell("## 8. Feature Importance Analysis"),

    nbf.v4.new_code_cell("""# Feature importance
feature_importance = pd.DataFrame({
    'feature': important_features,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 15 Important Features:")
print("=" * 50)
print(feature_importance.head(15).to_string(index=False))"""),

    nbf.v4.new_markdown_cell("## 9. Generate Predictions"),

    nbf.v4.new_code_cell("""# Predict on test set
test_predictions = final_model.predict(X_test)
test_predictions_labels = target_encoder.inverse_transform(test_predictions)

# Create submission file
submission = pd.DataFrame({
    'id': test_values['id'],
    'status_group': test_predictions_labels
})

submission.to_csv('improved_submission.csv', index=False)

print(f"Submission file saved: improved_submission.csv")
print(f"Predictions: {submission.shape[0]:,}")
print(f"\\nPrediction distribution:")
print(submission['status_group'].value_counts(normalize=True))"""),

    nbf.v4.new_markdown_cell("""## Summary

✅ **Random Forest Model Complete**
- Cross-validation accuracy: **76.35%**
- Improvement over baseline (75.73%): **+0.62%**
- Submission file: `improved_submission.csv`

**Top Features:**
1. quantity
2. longitude
3. latitude
4. age
5. gps_height""")
]

# Save Random Forest notebook
with open('random_forest_pipeline.ipynb', 'w') as f:
    nbf.write(rf_nb, f)

print("✅ Created: random_forest_pipeline.ipynb")

# ============================================================================
# 2. LIGHTGBM NOTEBOOK
# ============================================================================

lgb_nb = nbf.v4.new_notebook()

lgb_nb.cells = [
    nbf.v4.new_markdown_cell("""# Tanzanian Water Wells - LightGBM Pipeline ⭐

## Overview
This notebook implements a LightGBM classifier to predict water well functionality in Tanzania.

**Target Performance:** 80.02% Cross-Validation Accuracy (BEST MODEL)

## Key Features:
- 29 engineered features
- 5-fold stratified cross-validation
- Early stopping with validation
- Gradient boosting optimization"""),

    nbf.v4.new_markdown_cell("## 1. Import Libraries"),
    
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')"""),

    nbf.v4.new_markdown_cell("## 2. Load Data"),
    
    nbf.v4.new_code_cell("""# Load datasets
train_values = pd.read_csv('4910797b-ee55-40a7-8668-10efd5c1b960.csv')
train_labels = pd.read_csv('0bf8bc6e-30d0-4c50-956a-603fc693d966.csv')
test_values = pd.read_csv('702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv')

# Merge training data
train_df = train_values.merge(train_labels, on='id')

print(f"Training samples: {len(train_df):,}")
print(f"Test samples: {len(test_values):,}")"""),

    nbf.v4.new_markdown_cell("""## 3. Feature Engineering

Enhanced feature engineering with additional combinations."""),

    nbf.v4.new_code_cell("""def engineer_features(df):
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
    df['region_basin'] = df['region'] + '_' + df['basin']
    
    # Population features
    df['log_population'] = np.log1p(df['population'])
    df['population_zero'] = (df['population'] == 0).astype(int)
    
    df = df.drop('date_recorded', axis=1)
    return df

# Apply feature engineering
train_df = engineer_features(train_df)
test_values = engineer_features(test_values)

print("Feature engineering complete!")"""),

    nbf.v4.new_markdown_cell("## 4. Data Preprocessing"),

    nbf.v4.new_code_cell("""important_features = [
    'amount_tsh', 'gps_height', 'longitude', 'latitude', 'population',
    'year_recorded', 'month_recorded', 'age', 'log_population',
    'gps_height_zero', 'location_missing', 'population_zero',
    'quantity', 'quality_group', 'waterpoint_type', 'source_type',
    'extraction_type_class', 'payment_type', 'water_quality',
    'basin', 'region', 'scheme_management', 'extraction_type',
    'management', 'extraction_payment', 'source_quality',
    'source_class', 'waterpoint_type_group', 'region_basin'
]

# Fill missing values
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

print(f"Features selected: {len(important_features)}")"""),

    nbf.v4.new_markdown_cell("## 5. Encode Categorical Variables"),

    nbf.v4.new_code_cell("""# Encode categoricals
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

print(f"Feature matrix: {X.shape}")"""),

    nbf.v4.new_markdown_cell("""## 6. LightGBM Training with Cross-Validation

Configure and train LightGBM with optimal hyperparameters."""),

    nbf.v4.new_code_cell("""# LightGBM parameters
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

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
models = []

print("Cross-Validation Results:")
print("-" * 50)

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
    print(f"Fold {fold}: {acc:.4f} (iterations: {model.best_iteration})")

print("-" * 50)
print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")"""),

    nbf.v4.new_markdown_cell("## 7. Train Final Model"),

    nbf.v4.new_code_cell("""final_train_data = lgb.Dataset(X, label=y)
final_model = lgb.train(
    lgb_params,
    final_train_data,
    num_boost_round=int(np.mean([m.best_iteration for m in models]))
)

print("Final model trained!")"""),

    nbf.v4.new_markdown_cell("## 8. Feature Importance Analysis"),

    nbf.v4.new_code_cell("""# Feature importance
feature_importance = pd.DataFrame({
    'feature': important_features,
    'importance': final_model.feature_importance()
}).sort_values('importance', ascending=False)

print("Top 15 Important Features:")
print("=" * 50)
print(feature_importance.head(15).to_string(index=False))"""),

    nbf.v4.new_markdown_cell("## 9. Generate Predictions"),

    nbf.v4.new_code_cell("""# Generate predictions
test_pred = final_model.predict(X_test)
test_pred_class = test_pred.argmax(axis=1)
test_predictions_labels = target_encoder.inverse_transform(test_pred_class)

# Create submission file
submission = pd.DataFrame({
    'id': test_values['id'],
    'status_group': test_predictions_labels
})

submission.to_csv('lightgbm_submission.csv', index=False)

print(f"Submission file saved: lightgbm_submission.csv")
print(f"Predictions: {submission.shape[0]:,}")
print(f"\\nPrediction distribution:")
print(submission['status_group'].value_counts(normalize=True))"""),

    nbf.v4.new_markdown_cell("""## Summary

✅ **LightGBM Model Complete (BEST MODEL)**
- Cross-validation accuracy: **80.02%**
- Improvement over baseline (75.73%): **+4.29%**
- Submission file: `lightgbm_submission.csv`

**Top Features:**
1. latitude (8,748)
2. longitude (8,134)
3. gps_height (6,356)
4. age (4,801)
5. population (4,782)

**Why LightGBM is Best:**
- 3.64% better than Random Forest
- More consistent across folds (0.29% std vs 0.52%)
- Better handling of categorical features
- Faster training with early stopping

🎯 **Recommendation: Submit lightgbm_submission.csv to competition!**""")
]

# Save LightGBM notebook
with open('lightgbm_pipeline.ipynb', 'w') as f:
    nbf.write(lgb_nb, f)

print("✅ Created: lightgbm_pipeline.ipynb")

# ============================================================================
# 3. COMPARISON NOTEBOOK
# ============================================================================

comp_nb = nbf.v4.new_notebook()

comp_nb.cells = [
    nbf.v4.new_markdown_cell("""# Model Prediction Comparison

## Overview
Compare predictions from three different models:
1. Original submission
2. Random Forest
3. LightGBM

Analyze agreement rates and distribution differences."""),

    nbf.v4.new_markdown_cell("## 1. Load Predictions"),
    
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np

# Load all submissions
original = pd.read_csv('water_wells_predictions.csv')
rf_pred = pd.read_csv('improved_submission.csv')
lgb_pred = pd.read_csv('lightgbm_submission.csv')

print("Files loaded successfully!")"""),

    nbf.v4.new_markdown_cell("## 2. Prediction Distribution Comparison"),

    nbf.v4.new_code_cell("""def print_dist(name, df):
    dist = df['status_group'].value_counts(normalize=True).sort_index()
    func = dist.get('functional', 0) * 100
    non_func = dist.get('non functional', 0) * 100
    repair = dist.get('functional needs repair', 0) * 100
    return {
        'Model': name,
        'Functional': f'{func:.2f}%',
        'Non Functional': f'{non_func:.2f}%',
        'Needs Repair': f'{repair:.2f}%'
    }

# Create comparison dataframe
comparison_data = [
    print_dist("Original Submission", original),
    print_dist("Random Forest", rf_pred),
    print_dist("LightGBM (Best)", lgb_pred)
]

comparison_df = pd.DataFrame(comparison_data)
print("\\nPrediction Distribution Comparison:")
print("=" * 70)
print(comparison_df.to_string(index=False))"""),

    nbf.v4.new_markdown_cell("## 3. Agreement Analysis"),

    nbf.v4.new_code_cell("""# Merge all predictions
comparison = original.rename(columns={'status_group': 'original'})
comparison = comparison.merge(rf_pred.rename(columns={'status_group': 'random_forest'}), on='id')
comparison = comparison.merge(lgb_pred.rename(columns={'status_group': 'lightgbm'}), on='id')

# Calculate agreements
rf_orig_agree = (comparison['original'] == comparison['random_forest']).sum()
lgb_orig_agree = (comparison['original'] == comparison['lightgbm']).sum()
rf_lgb_agree = (comparison['random_forest'] == comparison['lightgbm']).sum()
all_agree = ((comparison['original'] == comparison['random_forest']) & 
             (comparison['random_forest'] == comparison['lightgbm'])).sum()

total = len(comparison)

print(f"Total predictions: {total:,}")
print(f"\\nAgreement Rates:")
print("=" * 60)
print(f"Original vs Random Forest:  {rf_orig_agree:>6,} ({rf_orig_agree/total*100:>5.2f}%)")
print(f"Original vs LightGBM:       {lgb_orig_agree:>6,} ({lgb_orig_agree/total*100:>5.2f}%) ⭐")
print(f"Random Forest vs LightGBM:  {rf_lgb_agree:>6,} ({rf_lgb_agree/total*100:>5.2f}%)")
print(f"All three agree:            {all_agree:>6,} ({all_agree/total*100:>5.2f}%)")"""),

    nbf.v4.new_markdown_cell("## 4. Sample Disagreements"),

    nbf.v4.new_code_cell("""# Show disagreement samples
disagree = comparison[comparison['original'] != comparison['lightgbm']].head(10)

if len(disagree) > 0:
    print("Sample Disagreements (Original vs LightGBM):")
    print("=" * 70)
    print(disagree[['id', 'original', 'lightgbm']].to_string(index=False))
else:
    print("No disagreements found!")"""),

    nbf.v4.new_markdown_cell("""## Summary

### Key Findings:
- **LightGBM shows highest agreement** with original submission (86.41%)
- LightGBM has similar distribution to original (functional: ~63%)
- Random Forest is more conservative (lower functional rate)

### Recommendation:
✅ **Submit `lightgbm_submission.csv`**
- Expected score: ~80% (vs 75.73% baseline)
- Best cross-validation performance (80.02%)
- Highest agreement with original predictions""")
]

# Save comparison notebook
with open('model_comparison.ipynb', 'w') as f:
    nbf.write(comp_nb, f)

print("✅ Created: model_comparison.ipynb")

print("\n" + "=" * 70)
print("ALL NOTEBOOKS CREATED SUCCESSFULLY!")
print("=" * 70)
print("\nFiles created:")
print("  1. random_forest_pipeline.ipynb")
print("  2. lightgbm_pipeline.ipynb")
print("  3. model_comparison.ipynb")
