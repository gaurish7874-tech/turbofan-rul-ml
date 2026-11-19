# FD001 training-only accuracy improvement

The nested selection procedure changed accuracy within 20 cycles by **+17.00 percentage points** and within 10 cycles by **+12.00 percentage points** on 300 simulated endpoints from 100 training engines.

Each endpoint follows 35%, 60%, or 85% of a training engine's recorded observations. Endpoints carry uncapped actual RUL. The original five engine folds are the outer assessment; three engine-grouped folds inside each outer training split choose a configuration. Highest within 20 cycles, then highest within 10 cycles, then lowest MAE. Every score for an outer validation engine comes from a model that excluded that engine from fitting and selection.

The saved candidate is **Sensor-history XGBoost**, chosen by the same ranking rule on a separate five-fold run using all training engines, and then fitted to every training row. The nested estimate applies to the selection procedure. The full-training selection table describes the model selection run.

Historical NASA test results stay separate. These simulated training endpoints can differ from the NASA endpoints in RUL distribution and do not provide a new NASA test accuracy estimate.
