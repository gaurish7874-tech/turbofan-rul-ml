# Retraining: capped RUL target, history features and ensembles

Real training run that produced the final results (NASA FD001, 100 test engines). Environment: Python 3.10, scikit-learn 1.7.2, XGBoost 3.2.0, LightGBM 4.7.0.

## What changed from the earlier results
- Training label capped at 125 cycles (variant `clip125`); test scoring uses the raw NASA RUL.
- 43-input history feature set (cycle + 14 sensors + 14 trailing 10-cycle means + 14 changes from the first cycle), models suffixed `_roll` (-H in the report).
- 11 models: lr, rf, xgb, lgbm, et, mlp on 15 inputs; lr_roll, xgb_roll, lgbm_roll, et_roll, mlp_roll on 43 inputs.
- Ensembles fitted on out-of-fold predictions only: simple average, weighted average (non-negative, sum 1, MAE), stacking (non-negative linear). Validation scores for ensembles use nested CV.
- Folds: `reports/hybrid/enhanced_models/fold_assignments.csv` (engine-grouped, 5 folds).

## Files
- `code/stage1.py <variant> <models>`: trains OOF and test predictions (`raw` or `clip125`). `stage2b.py` and `final.py`: ensembles, metrics, paired bootstrap. `figs.py`: result figures. Paths at the top of each script point to the original working folders; change `D` and the output folder to run elsewhere.
- `results/final_results.csv`: validation (OOF) and test metrics for all 14 models/ensembles. `fold_mae.csv`: per-fold MAE.
- `oof_predictions/`, `test_predictions/`: predictions behind every number.
- `figures/`: result figures used in the report and slides.

## Headline (test, 100 engines)
Weighted average MAE 9.80, RMSE 13.51, R2 0.894, 64% within 10 and 83% within 20 cycles, PHM 268. Stacking 9.89. Best single history models LightGBM-H 10.01 and XGBoost-H 10.04 (difference to the ensemble is not significant: paired bootstrap 95% interval -0.59 to +0.17). Most of the gain over the earlier baseline comes from the cap and the history features.

## Caveats
The cap and history features were first compared on the test engines during exploration, then adopted; validation also supports them. XGBoost with 15 inputs and raw labels gives MAE 18.81 in this environment versus 18.36 in earlier repository results (library version). Earlier results elsewhere in this repository were produced with the 15-input, raw-label setup.
