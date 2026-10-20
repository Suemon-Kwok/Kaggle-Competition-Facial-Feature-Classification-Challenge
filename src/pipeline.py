"""Shared data pipeline for ENGE707 Kaggle project (cat/dog facial feature classification).

This module is imported by every notebook (`import pipeline`) so that the training, validation and test data
all go through exactly the same steps. Importing it runs the module-level code below once: it reads the label
files, encodes the classes, creates the stratified train/validation split and defines the dataset builders.

What it provides:
    labels, train_df, val_df, test_df     tables of image ids, filenames and (for train/val) integer labels
    class_names, class_to_idx, idx_to_class   the alphabetical class <-> integer mapping
    TRAIN_DIR, TEST_DIR                   folders that hold the images
    make_dataset(...)                     batched tf.data pipeline for the training or validation images
    make_test_dataset()                   batched tf.data pipeline for the unlabelled test images
"""
from pathlib import Path  # Path: build file paths that work on any operating system

import pandas as pd  # pandas: read the CSV files and hold them as tables (DataFrames)
import tensorflow as tf  # TensorFlow: image decoding, augmentation layers and the tf.data input pipeline
from sklearn.model_selection import train_test_split  # scikit-learn helper that splits data into two random subsets

# --- Paths (relative to this file, so they work from any notebook) ---
ROOT = Path(__file__).resolve().parent.parent  # this file is <project>/src/pipeline.py: .parent = src, .parent.parent = <project>
DATA_DIR = ROOT / "data"  # folder that holds the competition data
TRAIN_DIR = DATA_DIR / "train" / "train"  # folder with the training images (the archive has a nested train/train folder)
TEST_DIR = DATA_DIR / "test" / "test"  # folder with the test images (nested test/test folder)

# --- Settings ---
IMG_SIZE = 224  # every image is resized to 224 x 224 pixels, the input size of the pretrained networks
BATCH_SIZE = 32  # number of images processed together in one training step
SEED = 42  # one fixed random seed, so the split and the shuffling are repeatable

# --- Label encoding (alphabetical, reproducible) ---
labels = pd.read_csv(DATA_DIR / "train_labels.csv")  # label file: one row per training image (image_id, class)
class_names = sorted(labels["class"].unique())  # the eight class names in alphabetical order (CatEye ... DogNose)
class_to_idx = {name: i for i, name in enumerate(class_names)}  # dictionary: class name -> integer 0..7
idx_to_class = {i: name for name, i in class_to_idx.items()}  # reverse dictionary: integer -> class name
labels["label"] = labels["class"].map(class_to_idx)  # new column 'label': the integer code of each row's class
labels["filename"] = labels["image_id"].apply(lambda x: f"{x:05d}.jpg")  # image files use five-digit zero-padded names, e.g. 1 -> 00001.jpg

# --- Stratified 80/20 split ---
train_df, val_df = train_test_split(  # returns the training rows and the validation rows
    labels, test_size=0.2, stratify=labels["label"], random_state=SEED  # 20% for validation; same class proportions in both; fixed seed
)

# --- Test set (ids kept as in sample_submission.csv) ---
test_df = pd.read_csv(DATA_DIR / "sample_submission.csv")  # Kaggle's template: it lists every test image_id in the required order
test_df["filename"] = test_df["image_id"].apply(lambda x: f"{x:05d}.jpg")  # build the same zero-padded filenames for the test images

# --- Loading / preprocessing (identical for train, val and test) ---
def load_and_preprocess(filename, label):  # turns one (filename, label) pair into (image tensor, label)
    image = tf.io.read_file(str(TRAIN_DIR) + "/" + filename)  # read the raw bytes of the training image file
    image = tf.image.decode_jpeg(image, channels=3)  # decode the JPEG into a height x width x 3 (RGB) array
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])  # resize to 224 x 224 (small images are upscaled)
    return image / 255.0, label  # scale pixel values from 0-255 to 0-1 and pass the label through unchanged

def load_test_image(filename):  # same as above for the test set, which has no labels
    image = tf.io.read_file(str(TEST_DIR) + "/" + filename)  # read the raw bytes of the test image file
    image = tf.image.decode_jpeg(image, channels=3)  # decode the JPEG into an RGB array
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])  # resize to 224 x 224, identical to training
    return image / 255.0  # scale pixel values to 0-1, identical to training

# --- Augmentation (training only) ---
data_augmentation = tf.keras.Sequential([  # a chain of random image transformations
    tf.keras.layers.RandomFlip("horizontal"),  # randomly mirror the image left-right
    tf.keras.layers.RandomRotation(0.05),  # random rotation of up to 5% of a full turn (about +/-18 degrees)
    tf.keras.layers.RandomBrightness(0.1, value_range=(0.0, 1.0)),  # brightness change of up to 10%; pixels are in the range 0..1
    tf.keras.layers.RandomContrast(0.1),  # random contrast change of up to 10%
])

def augment(image, label):  # applies the augmentation to a batch of images; labels pass through unchanged
    image = data_augmentation(image, training=True)  # training=True switches the random layers on
    return tf.clip_by_value(image, 0.0, 1.0), label  # force pixel values back into 0..1 in case a transformation overshoots

# --- Dataset builders ---
def make_dataset(df, shuffle=False, augment_data=False):  # builds the batched pipeline for a train or validation DataFrame
    ds = tf.data.Dataset.from_tensor_slices((df["filename"].values, df["label"].values))  # dataset of (filename, label) pairs
    if shuffle:  # only the training set is shuffled
        ds = ds.shuffle(buffer_size=len(df), seed=SEED)  # random order using a buffer as large as the whole set; fixed seed
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)  # load, resize and scale the images in parallel
    ds = ds.batch(BATCH_SIZE)  # group 32 images into one batch
    if augment_data:  # only the training set is augmented
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)  # apply the random transformations to each batch
    return ds.prefetch(tf.data.AUTOTUNE)  # prepare the next batch while the GPU works on the current one

def make_test_dataset():  # builds the batched pipeline for the test images (no labels, no shuffling, no augmentation)
    ds = tf.data.Dataset.from_tensor_slices(test_df["filename"].values)  # dataset of test filenames, kept in the Kaggle order
    ds = ds.map(load_test_image, num_parallel_calls=tf.data.AUTOTUNE)  # load, resize and scale the images in parallel
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)  # batch of 32, with the next batch prepared in the background
