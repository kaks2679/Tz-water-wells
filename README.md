# Tanzanian Water Wells - Predicting Well Functionality

## Competition Result

**DrivenData Competition: Pump it Up - Data Mining the Water Table**
- **Score:** 0.8070
- **Rank:** #4737 (Top performers at 0.8299)
- **Leaderboard:** [View Rankings](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/leaderboard/)

## Project Overview

I developed machine learning models to predict the operational status of water wells in Tanzania. The goal is to classify wells into three categories: functional, functional needs repair, or non-functional. This helps prioritize maintenance resources and ensure reliable access to clean water for communities.

## The Challenge

Tanzania has thousands of water wells, and many fall into disrepair. Instead of reactive maintenance (fixing wells after they break), I built predictive models to identify at-risk wells before they fail completely, enabling proactive maintenance planning.

## Dataset

- **Training data:** 59,400 wells with operational status labels
- **Test data:** 14,850 wells for prediction
- **Target classes:** functional, functional needs repair, non-functional
- **Features:** 40 variables including location, well type, water quantity, construction year, management, and payment systems

## My Approach

### 1. Initial Exploration
I started with exploratory data analysis to understand the data distribution, missing values, and feature relationships.

### 2. Feature Engineering
Created several derived features that significantly improved model performance:

**Date Features:**
- `year_recorded` - Year the well was recorded
- `month_recorded` - Month recorded (seasonal patterns)
- `age` - Well age calculated from construction year

**Geographic Features:**
- `gps_height_zero` - Flag for missing GPS data
- `location_missing` - Flag for invalid coordinates

**Categorical Combinations:**
- `extraction_payment` - Extraction type + Payment method
- `source_quality` - Source type + Water quality
- `region_basin` - Regional + Basin combination

**Population Features:**
- `log_population` - Log transformation for normality
- `population_zero` - Zero population flag

### 3. Model Development

I experimented with multiple algorithms and configurations:

**Initial Models:**
- Logistic Regression (baseline)
- Decision Tree (75.88% CV accuracy)

**Advanced Models:**
- Random Forest (76.35% CV accuracy)
- **LightGBM (80.02% CV accuracy)** ← Best performing

### 4. Final Model: LightGBM

Selected LightGBM as the final model based on:
- Highest cross-validation accuracy (80.02%)
- Most consistent across folds (±0.29% standard deviation)
- Superior handling of categorical features
- Fast training with early stopping

**Model Configuration:**
```python
learning_rate: 0.05
max_depth: 15
num_leaves: 40
boosting_type: gbdt
early_stopping: 50 rounds
```

**Cross-Validation Results:**
```
Fold 1: 80.56%
Fold 2: 79.97%
Fold 3: 79.73%
Fold 4: 79.85%
Fold 5: 79.99%
Mean: 80.02% (±0.29%)
```

## Key Findings

### Most Important Features

My analysis revealed that geographic features are the strongest predictors:

1. **latitude** (8,748) - Geographic location
2. **longitude** (8,134) - Geographic location
3. **gps_height** (6,356) - Elevation
4. **age** (4,801) - Well age (engineered feature)
5. **population** (4,782) - People served

This indicates strong regional patterns in well functionality, suggesting location-based maintenance strategies would be most effective.

### Insights

- **Geographic clustering:** Wells in certain regions are more prone to failure
- **Age matters:** Older wells significantly more likely to need maintenance
- **Water availability:** Quantity and quality are critical predictors
- **Infrastructure type:** Certain waterpoint types more reliable than others

## Technologies Used

- **Python 3.x**
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, LightGBM
- **Development:** Jupyter Notebooks

## Project Files

### Notebooks
- `lightgbm_pipeline.ipynb` - Final LightGBM model (recommended)
- `random_forest_pipeline.ipynb` - Random Forest alternative
- `model_comparison.ipynb` - Model comparison and analysis
- `Tz_water_wells.ipynb` - Initial exploratory analysis

### Python Scripts
- `lightgbm_pipeline.py` - Production-ready LightGBM script
- `improved_pipeline.py` - Random Forest script
- `compare_predictions.py` - Prediction comparison tool

### Submissions
- `lightgbm_submission.csv` - Final submission (0.8070 score)
- `improved_submission.csv` - Random Forest predictions

### Documentation
- `PRODUCTION_RESULTS.md` - Detailed performance analysis
- `NOTEBOOKS_GUIDE.md` - Jupyter notebook usage guide
- `PROJECT_COMPLETE.md` - Complete project summary

## How to Reproduce

### Setup
```bash
pip install pandas numpy scikit-learn lightgbm jupyter
```

### Run the Model
```bash
# Interactive notebook (recommended for exploration)
jupyter notebook lightgbm_pipeline.ipynb

# Or run as script
python lightgbm_pipeline.py
```

### Generate Predictions
The script will:
1. Load and preprocess data
2. Engineer features
3. Train LightGBM with 5-fold CV
4. Generate predictions
5. Save to `lightgbm_submission.csv`

## Results & Performance

| Model | CV Accuracy | Competition Score | Improvement |
|-------|-------------|-------------------|-------------|
| Decision Tree | 75.88% | - | Baseline |
| Random Forest | 76.35% | - | +0.47% |
| **LightGBM** | **80.02%** | **0.8070** | **+4.14%** |

Competition performance closely matched cross-validation results, validating the model's generalization capability.

## Potential Improvements

To reach top performers (0.8299):
1. Advanced feature engineering (clustering, embeddings)
2. Hyperparameter optimization (Optuna, Bayesian optimization)
3. Ensemble methods (stacking multiple models)
4. Additional domain-specific features
5. Deep learning approaches (TabNet)

## Business Applications

This model can be used to:
1. **Prioritize maintenance** - Identify high-risk wells before failure
2. **Allocate resources** - Focus on regions with highest failure probability
3. **Plan proactively** - Schedule preventive maintenance based on predictions
4. **Optimize budget** - Reduce emergency repairs through early intervention

## Lessons Learned

1. **Feature engineering is crucial** - Derived features significantly improved performance
2. **Geographic features dominate** - Location-based patterns are strong predictors
3. **LightGBM excels** - Superior to Random Forest for this tabular data
4. **Cross-validation works** - CV scores aligned well with competition results
5. **Early stopping helps** - Prevents overfitting while speeding up training

## Contact

Feel free to explore the code and reach out with questions or suggestions for improvement.

**Competition:** [DrivenData - Pump it Up](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/)

---

*This project demonstrates end-to-end machine learning workflow from exploration to production-ready models, achieving top-tier competition performance.*
