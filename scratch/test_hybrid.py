import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from src.evaluate import compute_regression_metrics

rf_oof = pd.read_csv('reports/hybrid/enhanced_models/random_forest/oof_predictions.csv')
xgb_oof = pd.read_csv('reports/hybrid/enhanced_models/xgboost/oof_predictions.csv')
base_rf_oof = pd.read_csv('reports/individual_models/random_forest/oof_predictions.csv')
base_xgb_oof = pd.read_csv('reports/individual_models/xgboost/oof_predictions.csv')

y_true = rf_oof['actual_RUL'].values

print('--- Enhanced RF + XGB Ensembles ---')
for w_xgb in np.linspace(0, 1, 11):
    pred = (1 - w_xgb) * rf_oof['predicted_RUL'].values + w_xgb * xgb_oof['predicted_RUL'].values
    m = compute_regression_metrics(y_true, pred)
    print(f"Weight XGB {w_xgb:.2f} | MAE: {m['MAE']:.2f}, RMSE: {m['RMSE']:.2f}, R2: {m['R2']:.4f}")

print('\n--- Hybrid Blends (Baseline Best + Enhanced Best) ---')
# Baseline best is 0.70 XGB + 0.30 RF (MAE: 26.54, RMSE: 37.65, R2: 0.7012)
base_blend = 0.70 * base_xgb_oof['predicted_RUL'].values + 0.30 * base_rf_oof['predicted_RUL'].values
m_base = compute_regression_metrics(y_true, base_blend)
print(f"Baseline Weighted Ensemble: MAE: {m_base['MAE']:.2f}, RMSE: {m_base['RMSE']:.2f}, R2: {m_base['R2']:.4f}")

# Best enhanced ensemble
enh_best = 0.40 * rf_oof['predicted_RUL'].values + 0.60 * xgb_oof['predicted_RUL'].values
m_enh = compute_regression_metrics(y_true, enh_best)
print(f"Enhanced Weighted Ensemble: MAE: {m_enh['MAE']:.2f}, RMSE: {m_enh['RMSE']:.2f}, R2: {m_enh['R2']:.4f}")

for alpha in np.linspace(0, 1, 21):
    hyb = (1 - alpha) * base_blend + alpha * enh_best
    m = compute_regression_metrics(y_true, hyb)
    print(f"Alpha {alpha:.2f} (Enhanced share) | MAE: {m['MAE']:.2f}, RMSE: {m['RMSE']:.2f}, R2: {m['R2']:.4f}")
