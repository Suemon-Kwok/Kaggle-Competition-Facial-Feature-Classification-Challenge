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
| 11 | EfficientNetB0 stage 2 with AdamW (weight decay 0.05) + label smoothing 0.1, seed 42 (from the stage-1 checkpoint of #1) | 0.9953 | - | 60 | not submitted |
| 12 | Same as #11, seed 7 | 0.9953 | - | 60 | not submitted |
| 13 | Average of #11 and #12 (baseline pair #2 + #7: 66 errors) | 0.9955 | - | 57 | not submitted |
| 14 | Equal-weight average of 6 models (#2, #4, #7, #8, #11, #12) | 0.9965 | - | 45 | 0.9800 (v4b; submitted although above the pre-set rule of below 43) |
| 15 | ResNet50V2 stage 2 with AdamW (weight decay 0.05) + label smoothing 0.1, seed 42 (from the stage-1 checkpoint of #3) | 0.9950 | - | 64 | not submitted |
| 16 | Same as #15, seed 7 | 0.9954 | - | 59 | not submitted |
| 17 | Average of #15 and #16 (baseline pair #4 + #8: 59 errors) | 0.9957 | - | 55 | not submitted |
| 18 | Equal-weight average of 4 regularised models (#11, #12, #15, #16) | 0.9973 | - | 35 | 0.9784 (v4a) |
| 19 | EfficientNetV2-S, two-stage fine-tune (last 20% of layers unfrozen, seed 42) | 0.9899 | - | 129 | not submitted |
| 20 | ConvNeXt-Tiny, same recipe | 0.9924 | - | 97 | not submitted |
| 21 | Equal-weight average of 10 models (#2, #4, #7, #8, #11, #12, #15, #16, #19, #20) | 0.9971 | - | 37 | 0.9804 (v5; exploratory, outside pre-set rule) |

Notes: rows 7-9 are a clean rerun of the full notebook as a reproducibility check
(validation accuracy computed from the error counts). Compared with rows 2, 4 and 6,
single models varied by about +/-10 validation errors, and the ensemble was the best
entry in both runs. The rerun changed 331 (v1) and 233 (v2) of 29,443 test predictions.

Row 10: variance-reduction experiment (no new training). It changed 107 of 29,443 test predictions relative to v2 (row 6).
Rows 11-14: regularisation experiment. Regularised runs repeated at 60 errors versus 73 and 75 for the baselines; the 6-model ensemble did not beat the 43-error 4-model ensemble.
Rows 15-18: regularised ResNet50V2 runs (64 and 59 errors versus 63 and 75 for the baselines) and the 4-model regularised ensemble (35 errors, pre-set rule of at most 38 met; Kaggle 0.9784, v4a). The 6-model ensemble (row 14) was submitted although above its pre-set rule and scored 0.9800 (v4b); validation errors and public score did not rank the ensembles the same way.

Rows 19-21: further backbones (EfficientNetV2-S 129 errors, ConvNeXt-Tiny 97 errors) were weaker than the existing single models (59-75) and no ensemble that included them had fewer than 35 errors. The 10-model average (row 21, 37 errors) missed its pre-set rule of at most 30 but was submitted as an exploratory entry (v5) and scored 0.9804; v4b and v4a remain the selected submissions.
