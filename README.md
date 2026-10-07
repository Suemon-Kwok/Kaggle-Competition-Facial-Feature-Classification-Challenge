# ENGE707 Assessment 3 - Facial Feature Classification Challenge

Individual Kaggle project: 8-class classification of cat and dog facial features
(CatEye, DogEye, CatFace, DogFace, CatMouth, DogMouth, CatNose, DogNose).

- Kaggle username: kwoksuemon (leaderboard name: Suemon Kwok)
- Final approach: ensemble (probability average) of fine-tuned EfficientNetB0 and ResNet50V2
- Kaggle public leaderboard: 0.9786 (v2 ensemble); 0.9767 (v1 EfficientNetB0 alone)
- Validation accuracy (stratified 80/20 split): 0.9964 ensemble; see docs/experiment_log.md

## Repository layout

| Path | Contents |
|---|---|
| notebooks/01_eda.ipynb | Task 1: dataset exploration |
| notebooks/02_preprocessing.ipynb | Task 2: preprocessing and pipeline development |
| notebooks/03_model.ipynb | Tasks 3-6: models, training, evaluation, ensemble, submission files |
| src/pipeline.py | Shared data pipeline (split, labels, loading, augmentation) |
| docs/experiment_log.md | Table of all experiments and scores |
| figures/ | Learning curves, confusion matrix, misclassified examples |
| submissions/ | Kaggle submission CSVs (v1, v2) |
| requirements.txt | Exact package versions used |

Model checkpoints (models/) and the data are not stored in the repository.

## How to reproduce

1. Create a Python 3.12 virtual environment and install dependencies:
   `pip install -r requirements.txt`
2. Download the competition data from the ENGE707 Kaggle competition
   (private, invitation only) and place it so these paths exist:
   `data/train/train/`, `data/test/test/`, `data/train_labels.csv`, `data/sample_submission.csv`
3. Run the notebooks top to bottom in order (01, 02, 03) after a kernel restart.
   An NVIDIA GPU is strongly recommended (developed on an RTX 3080;
   training takes roughly 10 minutes per model).
4. 03_model.ipynb writes the submission files to submissions/.

## Notes

- Random seeds are fixed (split seed 42, `tf.keras.utils.set_random_seed(42)`),
  but GPU training is not fully deterministic, so small run-to-run
  differences are expected.
- Pretrained ImageNet weights are downloaded automatically by Keras.