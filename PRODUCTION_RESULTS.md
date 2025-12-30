# Production Pipeline Results

## Summary
Successfully implemented two production-grade machine learning pipelines that significantly outperform the baseline model.

## Performance Comparison

| Model | CV Accuracy | Improvement | Status |
|-------|-------------|-------------|--------|
| Initial Decision Tree (Notebook) | 75.88% | Baseline | ✓ Complete |
| Competition Submission Score | 75.73% | - | ✓ Submitted |
| **Random Forest Pipeline** | **76.35%** | **+0.62%** | ✓ Ready |
| **LightGBM Pipeline** ⭐ | **80.02%** | **+4.29%** | ✓ **RECOMMENDED** |

## Model Details

### 1. Random Forest Pipeline (`improved_pipeline.py`)

**Configuration:**
- 300 trees for final model (200 for CV)
- Max depth: 20
- Class-balanced weighting
- Sqrt feature selection
- Parallel processing

**Performance:**
- 5-Fold CV Mean: 76.35% ± 0.52%
- Fold scores: 77.33%, 76.20%, 76.06%, 76.30%, 75.83%

**Top Features:**
1. quantity (13.63%)
2. longitude (12.83%)
3. latitude (10.91%)
4. age (5.95%)
5. gps_height (5.87%)

### 2. LightGBM Pipeline (`lightgbm_pipeline.py`) ⭐ RECOMMENDED

**Configuration:**
- Gradient boosting with early stopping
- Learning rate: 0.05
- Max depth: 15
- 40 leaves per tree
- L1/L2 regularization
- Bagging: 80% features, 80% data

**Performance:**
- 5-Fold CV Mean: **80.02% ± 0.29%**
- Fold scores: 80.56%, 79.97%, 79.73%, 79.85%, 79.99%
- Average iterations: ~499 (with early stopping)

**Top Features (by importance):**
1. latitude (8,748)
2. longitude (8,134)
3. gps_height (6,356)
4. age (4,801)
5. population (4,782)
6. extraction_payment (2,537)
7. quantity (2,161)

## Feature Engineering Applied

### Date Features
- `year_recorded` - Year well was recorded
- `month_recorded` - Month well was recorded
- `age` - Well age (year_recorded - construction_year)

### Geographic Features
- `gps_height_zero` - Flag for missing GPS height
- `location_missing` - Flag for missing lat/lon (0,0)

### Categorical Combinations
- `extraction_payment` - Extraction type + Payment type
- `source_quality` - Source type + Water quality
- `region_basin` - Region + Basin combination

### Population Features
- `log_population` - Log transformation of population
- `population_zero` - Flag for zero population

## Submission Files

| File | Model | Size | Status |
|------|-------|------|--------|
| `improved_submission.csv` | Random Forest | 291 KB | Ready |
| `lightgbm_submission.csv` | LightGBM | 270 KB | **RECOMMENDED** |

## Prediction Distribution Comparison

| Status | Training | Random Forest | LightGBM |
|--------|----------|---------------|----------|
| functional | 54.31% | 51.31% | 62.85% |
| non functional | 38.42% | 34.73% | 34.28% |
| functional needs repair | 7.27% | 13.96% | 2.87% |

## Usage

### Quick Start
```bash
# Run Random Forest pipeline
python improved_pipeline.py

# Run LightGBM pipeline (recommended)
python lightgbm_pipeline.py
```

### Submit to Competition
1. Upload `lightgbm_submission.csv` to competition platform
2. Expected score: ~80% (based on CV accuracy)
3. Compare with baseline score: 75.73%

## Key Insights

1. **Geographic Features Dominate**: Latitude, longitude, and GPS height are consistently the most important features across both models.

2. **Age Matters**: Well age (derived from construction year and recording date) is a strong predictor of functionality.

3. **Feature Combinations Work**: Creating interaction features like `extraction_payment` and `source_quality` improves model performance.

4. **LightGBM Superiority**: 
   - 3.64% absolute improvement over Random Forest
   - More consistent across folds (lower std dev)
   - Better handling of categorical features
   - Faster training time

## Next Steps

1. ✅ **Submit LightGBM predictions** - Use `lightgbm_submission.csv`
2. **Monitor leaderboard** - Compare actual score with CV (80.02%)
3. **Feature iteration** - If needed, add more interaction features
4. **Ensemble approach** - Combine Random Forest + LightGBM predictions
5. **Hyperparameter tuning** - Fine-tune LightGBM parameters for marginal gains

## Technical Requirements

```bash
# Install dependencies
pip install pandas numpy scikit-learn lightgbm

# Run pipelines
python improved_pipeline.py      # ~90 seconds
python lightgbm_pipeline.py      # ~90 seconds
```

## Reproducibility

- Random seed: 42 (all models)
- Cross-validation: Stratified 5-fold
- Feature selection: 28-29 features
- Missing value strategy: Median (numeric), 'unknown' (categorical)
- Encoding: Label encoding for all categoricals

---

**Recommendation**: Submit `lightgbm_submission.csv` for the best expected performance (80.02% CV accuracy).
