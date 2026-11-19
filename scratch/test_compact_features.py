import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from src.splitting import create_engine_group_splits, get_fold_data
from src.evaluate import compute_regression_metrics

# Load enhanced data
df = pd.read_csv('data/processed/train_FD001_enhanced.csv')
y = df['RUL']
unit_ids = df['unit_number']

# Let's test a compact hybrid feature set:
# Original 15 features + top 10 rolling features that scored highest in importance:
top_rolling = [
    'sensor_15_roll_mean_10',
    'sensor_4_roll_mean_5',
    'sensor_11_roll_mean_5',
    'sensor_15_roll_mean_5',
    'sensor_4_roll_mean_10',
    'sensor_2_roll_mean_10',
    'sensor_3_roll_mean_10',
    'sensor_21_roll_mean_10',
    'sensor_9_roll_mean_10',
    'sensor_14_roll_mean_10',
]
base_features = [
    'time_cycles', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7',
    'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13',
    'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'
]

compact_features = base_features + [f for f in top_rolling if f not in base_features]
print(f"Compact feature count: {len(compact_features)}")

X_compact = df[compact_features]
splits = create_engine_group_splits(X_compact, y, unit_ids, n_splits=5, verbose=False)

xgb_preds = np.zeros(len(df))
for f_idx, (tr_idx, val_idx) in enumerate(splits):
    fd = get_fold_data(X_compact, y, unit_ids, tr_idx, val_idx)
    xgb = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    xgb.fit(fd['X_train'], fd['y_train'])
    xgb_preds[val_idx] = xgb.predict(fd['X_val'])

m_xgb = compute_regression_metrics(y, xgb_preds)
print(f"Compact Feature XGBoost (25 features): MAE: {m_xgb['MAE']:.2f}, RMSE: {m_xgb['RMSE']:.2f}, R2: {m_xgb['R2']:.4f}")
