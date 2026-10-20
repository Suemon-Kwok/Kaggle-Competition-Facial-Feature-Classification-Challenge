# ENGE707 Assessment 3 - Facial Feature Classification Challenge

Individual Kaggle project: 8-class classification of cat and dog facial features
(CatEye, DogEye, CatFace, DogFace, CatMouth, DogMouth, CatNose, DogNose).

- Kaggle username: kwoksuemon (leaderboard name: Suemon Kwok)
- Best public leaderboard score: 0.9800 (v4b, six-model ensemble); other scores: 0.9767 (v1 EfficientNetB0), 0.9786 (v2 ensemble), 0.9790 (v3, four-model ensemble), 0.9784 (v4a, four regularised models)
- Final selected submissions: v4b and v4a
- Approach: fine-tuned EfficientNetB0 and ResNet50V2 (two-stage transfer learning), combined by probability averaging
- Validation accuracy (stratified 80/20 split, 12,732 images): 0.9964 (v2 ensemble), 0.9973 (v4a); full results in docs/experiment_log.md

## Repository layout
| Path | Contents |
|---|---|
| notebooks/01_eda.ipynb | Task 1: dataset exploration |
| notebooks/02_preprocessing.ipynb | Task 2: preprocessing and pipeline development |
| notebooks/03_model.ipynb | Tasks 3-6: models, training, evaluation, ensemble, submission files (run 1) |
| notebooks/03_model_rerun.ipynb | Clean rerun with identical code (run-to-run variation) |
| notebooks/04_regularisation_experiment.ipynb | Weight decay and label smoothing experiment, extra ensembles |
| notebooks/05_leakage_controlled_validation.ipynb | Validation re-scored without duplicate/near-duplicate images |
| src/pipeline.py | Shared data pipeline (split, labels, loading, augmentation) |
| docs/experiment_log.md | Table of all experiments and scores |
| figures/ | Learning curves, confusion matrices, misclassified examples |
| submissions/ | Kaggle submission CSVs (v1, v2, v3, v4a, v4b) |
| requirements.txt | Exact package versions used |

Submission files: v1 = submission_v1_effnetb0.csv, v2 = submission_v2_ensemble.csv, v3 = submission_v3_ensemble4.csv, v4a = submission_v4_reg_ensemble4.csv, v4b = submission_v4_ensemble6.csv.

Model checkpoints (models/) and the data are not stored in the repository.

## How to reproduce
1. Create a Python 3.12 virtual environment and install dependencies: `pip install -r requirements.txt`
2. Download the competition data (private, invitation only) so these paths exist:
   data/train/train/, data/test/test/, data/train_labels.csv, data/sample_submission.csv
3. Run the notebooks from the notebooks/ folder, restarting the kernel before each one, in this order
   (the notebooks depend on each other's files):
   1. 01_eda, then 02_preprocessing
   2. 03_model, Steps 1-25 (Step 25 archives the run-1 results as backup_run1)
   3. 03_model_rerun, all steps (creates backup_run2 and restores the run-1 files)
   4. 03_model, Steps 26-29 (need backup_run2)
   5. 04_regularisation_experiment
   6. 05_leakage_controlled_validation (needs docs/data_quality_check.csv from 01 and the validation probabilities saved by 03)
   An NVIDIA GPU is strongly recommended (developed on an RTX 3080; roughly 10-15 minutes per model).
4. 03_model and 04_regularisation_experiment write the submission files to submissions/.

## Notes
- Seeds are fixed (split seed 42, tf.keras.utils.set_random_seed(42)), but GPU training is not fully deterministic. Differences of about ten validation errors between runs are expected (see 03_model_rerun.ipynb).
- Pretrained ImageNet weights are downloaded automatically by Keras.
- Code comments were added after the report figures were captured, so most notebook screenshots in the report show uncommented code.
