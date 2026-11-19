# FD001 hybrid cross-validation

Enhanced equal average increased mean MAE by 0.69 cycles against Baseline equal average.

| Model | Mean MAE | Mean RMSE | Mean R² |
| --- | ---: | ---: | ---: |
| Enhanced equal average | 27.253 | 39.007 | 0.6789 |
| Baseline equal average | 26.558 | 37.699 | 0.7004 |
| Baseline XGBoost | 26.615 | 37.694 | 0.7005 |
| Baseline weighted average | 26.540 | 37.649 | 0.7012 |

**Selection criterion**  Lowest mean validation MAE across the five baseline folds, with mean RMSE as tiebreaker. The three enhanced candidates were Random Forest, XGBoost, and their fixed equal average. Each used the same 99 unscaled features. The estimators retained their baseline hyperparameters.

The historical weighted average used 30% Random Forest and 70% XGBoost. Its weight search used the same held-out folds used to report performance. The historical stack used global out-of-fold base predictions, so some meta-training inputs depended on outer validation engines. Both scores appear as historical references in the candidate comparison.

Each candidate prediction for an engine came from estimators fitted on other engines. The same cross-validation results were used for candidate selection and reporting. Metrics in the tables are unweighted means across folds. Plot annotations use pooled predictions.

The model manifest references the fitted C2 fold checkpoints. For inference, supply the observed history for each engine. The predictor averages each component across its fold models and applies the selected component weights.
