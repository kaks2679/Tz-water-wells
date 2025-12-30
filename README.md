# Tanzanian Water Wells: Predictive Maintenance for Clean Water Access

## Project Overview

This project uses machine learning to predict the operational status of water wells in Tanzania. By identifying which wells are functional, need repair, or are non-functional, we help stakeholders optimize maintenance efforts and ensure reliable access to clean water for communities.

## Business Problem

Tanzania faces significant challenges with water well functionality. Many wells fall into disrepair, leaving communities without access to clean water. This project addresses the inefficiency of reactive maintenance by building predictive models to identify at-risk wells before they fail completely.

## Stakeholders

- Tanzanian Ministry of Water
- International NGOs funding water infrastructure
- Local communities dependent on functional water sources

## Dataset

- Training data: 59,400 wells with features and operational status labels
- Test data: 14,850 wells for prediction
- Target variable: 3 classes (functional, functional needs repair, non-functional)
- Features: Mix of numeric and categorical variables including location, well type, water quantity, construction year, and management details

## Project Structure

The notebook follows a complete data science workflow:

1. **Business Understanding** - Defining stakeholders and objectives
2. **Data Understanding** - Exploring dataset structure, distributions, and missing values
3. **Data Preparation** - Cleaning, encoding, and feature engineering
4. **Modeling** - Building and tuning classification models
5. **Model Evaluation** - Comprehensive performance analysis
6. **Feature Importance** - Identifying key drivers of well functionality
7. **Predictions** - Generating predictions for test set

## Models Used

**Logistic Regression**
- Baseline and tuned versions
- Regularization parameter tuning (C values)

**Decision Tree**
- Baseline and tuned versions  
- Hyperparameter tuning (max depth, min samples split/leaf)

## Key Results

**Best Model: Tuned Decision Tree**
- Validation Accuracy: 75.88%
- Weighted F1-Score: 74.29%
- Best balance between performance and capturing non-linear patterns

## Key Findings

**Most Important Features:**
1. Quantity (water availability)
2. Waterpoint type (infrastructure design)
3. Geographic location (longitude, latitude, region)
4. Age and construction year
5. Management and payment systems

These features indicate that water availability, infrastructure type, and location are the strongest predictors of well functionality.

## Technologies Used

- Python
- Pandas, NumPy for data manipulation
- Matplotlib, Seaborn for visualization
- Scikit-learn for machine learning
- Classification models: Logistic Regression, Decision Tree

## Business Recommendations

**1. Predictive Maintenance Program**
Deploy the model to identify at-risk wells and prioritize preventive maintenance before complete failure occurs.

**2. Geographic Targeting Strategy**
Focus resources on high-risk regions identified by the model. Create regional risk maps to allocate maintenance teams efficiently.

**3. Enhanced Data Collection**
Improve accuracy of critical features like water quantity, waterpoint type, and location data to strengthen model reliability.

**4. Monitor High-Impact Features**
Pay special attention to wells with declining water quantity, older infrastructure, and specific waterpoint types prone to failure.

## Business Impact

**Missed Failures (High Risk):**
When non-functional wells are predicted as functional, communities lose water access and face emergency repair costs.

**False Alarms (Moderate Risk):**
When functional wells are predicted as needing repair, resources are wasted but water access continues.

The model helps reduce both risks by enabling proactive maintenance planning.

## How to Use

1. Load the training and test datasets
2. Run data preprocessing and feature engineering steps
3. Train the tuned Decision Tree model with optimal parameters
4. Generate predictions for new wells
5. Use predictions to prioritize maintenance schedules

## Files

- `Tz_water_wells.ipynb` - Main analysis notebook
- Training and test CSV files
- Submission format for predictions