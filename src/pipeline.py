"""Shared data pipeline for ENGE707 Kaggle project (cat/dog facial feature classification)."""
from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# --- Paths (relative to this file, so they work from any notebook) ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train" / "train"
TEST_DIR = DATA_DIR / "test" / "test"

# --- Settings ---
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42

# --- Label encoding (alphabetical, reproducible) ---
labels = pd.read_csv(DATA_DIR / "train_labels.csv")
class_names = sorted(labels["class"].unique())
class_to_idx = {name: i for i, name in enumerate(class_names)}
idx_to_class = {i: name for name, i in class_to_idx.items()}
labels["label"] = labels["class"].map(class_to_idx)
labels["filename"] = labels["image_id"].apply(lambda x: f"{x:05d}.jpg")

# --- Stratified 80/20 split ---
train_df, val_df = train_test_split(
    labels, test_size=0.2, stratify=labels["label"], random_state=SEED
)

# --- Test set (ids kept as in sample_submission.csv) ---
test_df = pd.read_csv(DATA_DIR / "sample_submission.csv")
test_df["filename"] = test_df["image_id"].apply(lambda x: f"{x:05d}.jpg")

# --- Loading / preprocessing (identical for train, val and test) ---
def load_and_preprocess(filename, label):
    image = tf.io.read_file(str(TRAIN_DIR) + "/" + filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image / 255.0, label

def load_test_image(filename):
    image = tf.io.read_file(str(TEST_DIR) + "/" + filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image / 255.0

# --- Augmentation (training only) ---
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomBrightness(0.1, value_range=(0.0, 1.0)),
    tf.keras.layers.RandomContrast(0.1),
])

def augment(image, label):
    image = data_augmentation(image, training=True)
    return tf.clip_by_value(image, 0.0, 1.0), label

# --- Dataset builders ---
def make_dataset(df, shuffle=False, augment_data=False):
    ds = tf.data.Dataset.from_tensor_slices((df["filename"].values, df["label"].values))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

def make_test_dataset():
    ds = tf.data.Dataset.from_tensor_slices(test_df["filename"].values)
    ds = ds.map(load_test_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
