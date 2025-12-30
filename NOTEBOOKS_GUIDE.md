# Jupyter Notebooks Guide

## Overview
This directory contains three production-ready Jupyter notebooks for the Tanzanian Water Wells prediction project.

## Notebooks

### 1. 🌲 random_forest_pipeline.ipynb
**Purpose:** Random Forest implementation with comprehensive feature engineering

**Performance:**
- Cross-validation accuracy: 76.35%
- Improvement over baseline: +0.62%

**Contents:**
- Data loading and exploration
- Feature engineering (date, geographic, categorical combinations)
- Data preprocessing and encoding
- 5-fold cross-validation
- Feature importance analysis
- Prediction generation

**Output:** `improved_submission.csv`

**Runtime:** ~90 seconds

---

### 2. ⭐ lightgbm_pipeline.ipynb (RECOMMENDED)
**Purpose:** LightGBM gradient boosting implementation (BEST MODEL)

**Performance:**
- Cross-validation accuracy: **80.02%**
- Improvement over baseline: **+4.29%**

**Contents:**
- Enhanced feature engineering (29 features)
- LightGBM with early stopping
- Robust cross-validation
- Detailed feature importance
- Optimized hyperparameters

**Output:** `lightgbm_submission.csv` ⭐

**Runtime:** ~90 seconds

**Key Advantages:**
- 3.64% better than Random Forest
- More consistent across folds (±0.29% std)
- Superior categorical feature handling
- Faster training with early stopping

---

### 3. 📊 model_comparison.ipynb
**Purpose:** Compare predictions from all three models

**Features:**
- Prediction distribution comparison
- Agreement rate analysis
- Sample disagreement inspection
- Visual comparison tables

**Insights:**
- LightGBM shows 86.41% agreement with original
- Identifies where models diverge
- Validates model consistency

**Runtime:** <5 seconds

## Quick Start

### Run Individual Notebooks

```bash
# Option 1: Jupyter Notebook
jupyter notebook random_forest_pipeline.ipynb
jupyter notebook lightgbm_pipeline.ipynb
jupyter notebook model_comparison.ipynb

# Option 2: Jupyter Lab
jupyter lab

# Option 3: VS Code
# Open .ipynb files directly in VS Code
```

### Run All Cells Programmatically

```bash
# Install jupyter if needed
pip install jupyter nbconvert

# Execute notebooks
jupyter nbconvert --to notebook --execute random_forest_pipeline.ipynb
jupyter nbconvert --to notebook --execute lightgbm_pipeline.ipynb
jupyter nbconvert --to notebook --execute model_comparison.ipynb
```

## Dependencies

```bash
pip install pandas numpy scikit-learn lightgbm jupyter
```

## Notebook Structure

All notebooks follow a consistent structure:

1. **Markdown Introduction** - Overview and objectives
2. **Library Imports** - Required packages
3. **Data Loading** - Load train/test datasets
4. **Feature Engineering** - Create derived features
5. **Preprocessing** - Handle missing values, encoding
6. **Model Training** - Cross-validation and training
7. **Feature Importance** - Analyze key predictors
8. **Predictions** - Generate submission file
9. **Summary** - Results and recommendations

## Key Features

### Clean Output
- ✅ Minimal print statements
- ✅ Markdown explanations between code cells
- ✅ Clear section headers
- ✅ Professional formatting
- ✅ No excessive logging

### Reproducibility
- Fixed random seed (42)
- Same feature engineering across models
- Consistent preprocessing
- Documented parameters

### Documentation
- Markdown cells explain each step
- Clear variable names
- Commented code where needed
- Summary sections

## Comparison: Notebooks vs Python Scripts

| Feature | Notebooks | Python Scripts |
|---------|-----------|----------------|
| Interactive | ✅ Yes | ❌ No |
| Markdown docs | ✅ Yes | ❌ Limited |
| Output control | ✅ Cell-by-cell | ❌ All at once |
| Visualization | ✅ Inline | ❌ Separate |
| Debugging | ✅ Easy | ⚠️ Harder |
| Production | ⚠️ Needs conversion | ✅ Ready |

## Expected Results

### Random Forest (random_forest_pipeline.ipynb)
```
Cross-Validation Results:
------------------------------------------
Fold 1: 0.7733
Fold 2: 0.7620
Fold 3: 0.7606
Fold 4: 0.7630
Fold 5: 0.7583
------------------------------------------
Mean CV Accuracy: 0.7635 (±0.0052)
```

### LightGBM (lightgbm_pipeline.ipynb)
```
Cross-Validation Results:
--------------------------------------------------
Fold 1: 0.8056 (iterations: 498)
Fold 2: 0.7997 (iterations: 500)
Fold 3: 0.7973 (iterations: 500)
Fold 4: 0.7985 (iterations: 500)
Fold 5: 0.7999 (iterations: 499)
--------------------------------------------------
Mean CV Accuracy: 0.8002 (±0.0029)
```

## Submission Files Generated

After running the notebooks, you'll have:

1. `improved_submission.csv` - Random Forest predictions
2. `lightgbm_submission.csv` - LightGBM predictions ⭐ **SUBMIT THIS**

## Troubleshooting

### Issue: Kernel Dies During Training
**Solution:** Increase memory or reduce `n_estimators`

### Issue: LightGBM Not Found
**Solution:** `pip install lightgbm`

### Issue: Cells Take Too Long
**Solution:** Reduce cross-validation folds or use smaller `num_boost_round`

### Issue: Different Results Each Run
**Solution:** Verify `random_state=42` is set in all random operations

## Next Steps

1. ✅ Run `lightgbm_pipeline.ipynb` (RECOMMENDED)
2. 📤 Submit `lightgbm_submission.csv` to competition
3. 📊 Run `model_comparison.ipynb` to analyze predictions
4. 🎯 Monitor competition leaderboard

## Performance Summary

| Notebook | Model | CV Accuracy | Status |
|----------|-------|-------------|--------|
| random_forest_pipeline.ipynb | Random Forest | 76.35% | ✅ Complete |
| **lightgbm_pipeline.ipynb** | **LightGBM** | **80.02%** | ⭐ **BEST** |
| model_comparison.ipynb | Analysis | N/A | 📊 Tool |

## Additional Resources

- Original notebook: `Tz_water_wells.ipynb`
- Python scripts: `improved_pipeline.py`, `lightgbm_pipeline.py`
- Documentation: `PRODUCTION_RESULTS.md`, `FINAL_SUMMARY.txt`

---

**Recommendation:** Start with `lightgbm_pipeline.ipynb` for the best results!

🎯 **Expected Competition Score: ~80%** (vs 75.73% baseline)
