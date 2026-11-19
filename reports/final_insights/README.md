# FD001 final model comparison

## Evaluation

Cross-validation values are saved means across five engine-grouped folds scored on available validation cycles. The NASA evaluation uses one prediction at the last observed cycle for each of the 100 held-out engines. These metrics summarize different sets of observations and target distributions. The held-out rankings describe generalization on FD001. The training configurations and blend weights were fixed before test scoring.

## Strongest results

The lowest reported CV MAE was **Weighted Averaging** at **26.54 cycles**. The lowest NASA test MAE was **XGBoost** at **18.36 cycles**.

- Lowest test RMSE was XGBoost at 25.69 cycles.
- Highest test R² was XGBoost at 0.6179.
- Lowest NASA PHM score was Stacking at 6096.54. The score penalizes RUL overestimates more heavily. [PHM 2008 scoring reference](https://ntrs.nasa.gov/api/citations/20205001125/downloads/Run_to_Failure_Simulation_Under_Real_Flight_Conditions_Dataset.pdf)

See the lowest-MAE table for the individual, ensemble, and hybrid leaders in each evaluation.

## Averaging and stacking

- Simple Averaging had 0.06 cycles lower MAE in CV and 0.74 cycles higher MAE on the NASA test set compared with XGBoost.
- Weighted Averaging had 0.08 cycles lower MAE in CV and 0.43 cycles higher MAE on the NASA test set compared with XGBoost.
- Stacking had 0.47 cycles higher MAE in CV and 0.18 cycles higher MAE on the NASA test set compared with Simple Averaging.
- Stacking had 0.49 cycles higher MAE in CV and 0.49 cycles higher MAE on the NASA test set compared with Weighted Averaging.

The historical weighted CV score was reported on the folds used to choose its weights. The historical stacking CV used global out-of-fold base predictions, so some meta-training inputs depended on outer validation engines. The final stack learned its meta-model only from training-side, engine-held-out predictions and scored the separate NASA test engines once.

## Engineered features

- Enhanced Random Forest had 0.55 cycles higher MAE in CV and 0.60 cycles higher MAE on the NASA test set compared with Random Forest.
- Enhanced XGBoost had 1.15 cycles higher MAE in CV and 1.76 cycles higher MAE on the NASA test set compared with XGBoost.
- Final Hybrid Model had 0.69 cycles higher MAE in CV and 0.97 cycles higher MAE on the NASA test set compared with Simple Averaging.

The rolling summaries may smooth useful short-lived changes or add correlated inputs alongside the raw readings. These comparisons measure the combined feature set at the existing model settings.

## Error patterns and limitations

For XGBoost, the highest endpoint MAE occurred in the **100 or more cycles** actual-RUL range. Its MAE was **25.76** over **34 engines**, with a mean signed error of **+11.20 cycles**. Signed error is predicted minus actual RUL. A positive value means the model predicted too much remaining life.

Engine 15 accounted for **49.4%** of the Enhanced Random Forest PHM score. Its actual RUL was 83 cycles and the model predicted 195.7. Large endpoint errors can dominate the aggregate PHM score.

FD001 is a simulated benchmark and the test set contains a single censored endpoint per engine. Endpoint accuracy does not measure the consistency of predictions earlier in an engine's life. Similar readings at different latent degradation states and differences in engine life can make RUL difficult to distinguish. These are possible explanations rather than causal findings from the error tables. The reported rankings have sampling uncertainty, particularly when MAE differences are small.

## Technical conclusion

The strongest observed FD001 generalization by MAE came from **XGBoost**. Cross-validation and held-out performance are presented separately. The paired comparisons show where the ensemble or feature changes helped at the frozen settings, and the signed-error results show where maintenance timing would be most affected by overestimation.
