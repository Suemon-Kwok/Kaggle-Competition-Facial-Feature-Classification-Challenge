# Experiment log (ENGE707 Kaggle)

| # | Model / setup | Val acc | Val loss | Val errors | Kaggle public |
|---|---|---|---|---|---|
| 1 | EfficientNetB0 stage 1 (frozen, lr 1e-3, 5 epochs) | 0.9457 | 0.1517 | - | - |
| 2 | EfficientNetB0 stage 2 (top blocks unfrozen, lr 1e-4, ReduceLROnPlateau, 12 epochs) | 0.9943 | 0.0206 | 73 | 0.9767 (v1) |
| 3 | ResNet50V2 stage 1 (frozen, lr 1e-3, 5 epochs) | 0.9687 | 0.0966 | - | - |
| 4 | ResNet50V2 stage 2 (conv5 unfrozen, lr 1e-4, early stopped epoch 11, restored epoch 8) | 0.9951 | 0.0189 | 63 | not submitted |
| 5 | EfficientNetB0 + horizontal-flip TTA | 0.9943 | - | 73 | not submitted |
| 6 | Ensemble: average of #2 and #4 probabilities | 0.9964 | 0.0138 (log loss) | 46 | 0.9786 (v2) |
| 7 | Rerun of #2 (EfficientNetB0, identical code, seed 42 set) | 0.9941 | - | 75 | not submitted |
| 8 | Rerun of #4 (ResNet50V2, identical code, seed 42 set) | 0.9941 | - | 75 | not submitted |
| 9 | Rerun of #6 (ensemble of #7 and #8) | 0.9958 | - | 53 | not submitted |
| 10 | Equal-weight average of 4 models: runs 1 and 2 of EfficientNetB0 and ResNet50V2 (#2, #4, #7, #8) | 0.9966 | - | 43 | 0.9790 (v3) |

Notes: rows 7-9 are a clean rerun of the full notebook as a reproducibility check
(validation accuracy computed from the error counts). Compared with rows 2, 4 and 6,
single models varied by about +/-10 validation errors, and the ensemble was the best
entry in both runs. The rerun changed 331 (v1) and 233 (v2) of 29,443 test predictions.

Row 10: variance-reduction experiment (no new training). It changed 107 of 29,443 test predictions relative to v2 (row 6).