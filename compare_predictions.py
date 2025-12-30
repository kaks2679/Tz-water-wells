#!/usr/bin/env python3
"""
Quick comparison of model predictions
"""
import pandas as pd
import numpy as np

print("=" * 70)
print("MODEL PREDICTION COMPARISON")
print("=" * 70)

# Load submissions
original = pd.read_csv('water_wells_predictions.csv')
rf_pred = pd.read_csv('improved_submission.csv')
lgb_pred = pd.read_csv('lightgbm_submission.csv')

print("\n📊 PREDICTION DISTRIBUTION COMPARISON\n")
print(f"{'Model':<25} {'Functional':<15} {'Non Functional':<18} {'Needs Repair':<15}")
print("-" * 70)

def print_dist(name, df):
    dist = df['status_group'].value_counts(normalize=True).sort_index()
    func = dist.get('functional', 0) * 100
    non_func = dist.get('non functional', 0) * 100
    repair = dist.get('functional needs repair', 0) * 100
    print(f"{name:<25} {func:>6.2f}%{'':<8} {non_func:>6.2f}%{'':<11} {repair:>6.2f}%")

print_dist("Original Submission", original)
print_dist("Random Forest", rf_pred)
print_dist("LightGBM (Best)", lgb_pred)

# Agreement analysis
print("\n" + "=" * 70)
print("PREDICTION AGREEMENT ANALYSIS")
print("=" * 70)

# Merge all predictions
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

print(f"\nTotal predictions: {total}")
print(f"\nAgreement rates:")
print(f"  Original vs Random Forest:  {rf_orig_agree:>5} ({rf_orig_agree/total*100:>5.2f}%)")
print(f"  Original vs LightGBM:       {lgb_orig_agree:>5} ({lgb_orig_agree/total*100:>5.2f}%)")
print(f"  Random Forest vs LightGBM:  {rf_lgb_agree:>5} ({rf_lgb_agree/total*100:>5.2f}%)")
print(f"  All three agree:            {all_agree:>5} ({all_agree/total*100:>5.2f}%)")

# Show disagreement samples
print("\n" + "=" * 70)
print("SAMPLE DISAGREEMENTS (where models differ)")
print("=" * 70)

disagree = comparison[comparison['original'] != comparison['lightgbm']].head(10)
if len(disagree) > 0:
    print(f"\n{'ID':<10} {'Original':<25} {'LightGBM':<25}")
    print("-" * 70)
    for _, row in disagree.iterrows():
        print(f"{row['id']:<10} {row['original']:<25} {row['lightgbm']:<25}")
else:
    print("\nNo disagreements found!")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print("\n✅ Submit: lightgbm_submission.csv")
print("   Reason: 80.02% CV accuracy (vs 75.73% baseline)")
print("   Expected improvement: +4.29%")
print("\n" + "=" * 70)
