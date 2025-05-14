import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path
import platform
import numpy as np
import psutil
import io
import imageio
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.metrics import precision_score, recall_score
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

# Set up logging
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    white = "\x1b[37;20m"

    FORMAT = "[%(asctime)s] "
    FORMAT += "{%(filename)s:%(lineno)d} "
    FORMAT += "%(levelname)s "
    FORMAT += "- %(message)s"

    FORMATS = {
        logging.DEBUG: grey + FORMAT + reset,
        logging.INFO: white + FORMAT + reset,
        logging.WARNING: yellow + FORMAT + reset,
        logging.ERROR: red + FORMAT + reset,
        logging.CRITICAL: bold_red + FORMAT + reset,
    }

    def format(self, record):
        DATEFORMAT = "%y-%m-%d %H:%M:%S"
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt=DATEFORMAT)
        return formatter.format(record)

def astronet_logger(name):
    """Create a logger with custom formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create console handler with custom formatter
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(CustomFormatter())
    
    # Add handler to logger
    logger.addHandler(ch)
    return logger

log = astronet_logger(__file__)

# Visualization utilities
plt.rc("font", size=20)
plt.rc("figure", figsize=(15, 3))

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def find_optimal_batch_size(training_set_length: int) -> int:
    """Determine optimal batch size to use. Ideally leave a large remainder such that the GPU is
    full for most of the time.
    """
    if training_set_length < 10000:
        batch_size_list = [16, 32, 64]
    else:
        batch_size_list = [2048, 4096]
    ratios = []
    for batch_size in batch_size_list:
        remainder = training_set_length % batch_size
        if remainder == 0:
            batch_size = remainder
        else:
            ratios.append(batch_size / remainder)

    index, ratio = min(enumerate(ratios), key=lambda x: abs(x[1] - 1))
    return batch_size_list[index]

def lazy_load_plasticc_wZ(X, Z, y):
    """Create a TensorFlow dataset from numpy arrays with redshift information."""
    def generator():
        for x, z, L in zip(X, Z, y):
            yield ({"input_1": x, "input_2": z}, L)

    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_signature=(
            {
                "input_1": tf.type_spec_from_value(X[0]),
                "input_2": tf.type_spec_from_value(Z[0]),
            },
            tf.type_spec_from_value(y[0]),
        ),
    )
    return dataset

def lazy_load_plasticc_noZ(X, y):
    """Create a TensorFlow dataset from numpy arrays without redshift information."""
    def generator():
        for x, L in zip(X, y):
            yield (x, L)

    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_signature=(
            tf.type_spec_from_value(X[0]),
            tf.type_spec_from_value(y[0]),
        ),
    )
    return dataset

class WeightedLogLoss(tf.keras.losses.Loss):
    """Weighted log loss for PLAsTiCC dataset."""
    def __init__(self, name="weighted_log_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        wtable = np.sum(y_true, axis=0) / y_true.shape[0]
        yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)
        yc = tf.cast(yc, tf.float64)
        y_true = tf.cast(y_true, tf.float64)
        wtable = tf.cast(wtable, tf.float64)
        loss = -(
            tf.reduce_mean(
                tf.math.divide_no_nan(
                    tf.reduce_mean(y_true * tf.math.log(yc), axis=0), wtable
                )
            )
        )
        return loss

class DistributedWeightedLogLoss(tf.keras.losses.Loss):
    """Distributed version of weighted log loss for multi-GPU training."""
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name="weighted_log_loss"):
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true, y_pred):
        wtable = np.sum(y_true, axis=0) / y_true.shape[0]
        yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)
        yc = tf.cast(yc, tf.float64)
        y_true = tf.cast(y_true, tf.float64)
        wtable = tf.cast(wtable, tf.float64)
        loss = -(
            tf.reduce_mean(
                tf.math.divide_no_nan(
                    tf.reduce_mean(y_true * tf.math.log(yc), axis=0), wtable
                )
            )
        )
        return loss

class SGEBreakoutCallback(tf.keras.callbacks.Callback):
    """Callback to stop training if job runs too long."""
    def __init__(self, threshold=24):
        super(SGEBreakoutCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs={}):
        hrs = subprocess.run(
            f"qstat -j {os.environ.get('JOB_ID')} | grep 'cpu' | awk '{{print $3}}' | awk -F ':' '{{print $1}}' | awk -F  '=' '{{print $2}}'",
            check=True,
            capture_output=True,
            shell=True,
            text=True,
        ).stdout.strip()

        if int(hrs) > self.threshold:
            log.info("Stopping training...")
            self.model.stop_training = True

class Training(object):
    def __init__(
        self,
        architecture,
        dataset,
        redshift=None,
        fink=None,
        avocado=None,
        testset=None,
    ):
        self.architecture = architecture
        self.dataset = dataset
        self.redshift = redshift
        self.fink = fink
        self.avocado = avocado
        self.testset = testset

    def __call__(self):
        """Train a given architecture with, or without redshift, on either UGRIZY or GR passbands"""
        def build_label():
            UNIXTIMESTAMP = int(time.time())
            try:
                VERSION = (
                    subprocess.check_output(["git", "describe", "--always"])
                    .strip()
                    .decode()
                )
            except Exception:
                VERSION = "unknown"
            JOB_ID = os.environ.get("JOB_ID")
            LABEL = f"{UNIXTIMESTAMP}-{JOB_ID}-{VERSION}"
            return LABEL

        LABEL = build_label()
        checkpoint_path = Path(self.architecture) / "models" / self.dataset / "checkpoints" / f"checkpoint-{LABEL}"
        csv_logger_file = Path("logs") / self.architecture / f"training-{LABEL}.log"

        # Create necessary directories
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        csv_logger_file.parent.mkdir(parents=True, exist_ok=True)

        # Lazy load data
        data_dir = Path("data/plasticc/processed")
        X_train = np.load(data_dir / "X_train.npy", mmap_mode="r")
        Z_train = np.load(data_dir / "Z_train.npy", mmap_mode="r")
        y_train = np.load(data_dir / "y_train.npy", mmap_mode="r")

        X_test = np.load(data_dir / "X_test.npy", mmap_mode="r")
        Z_test = np.load(data_dir / "Z_test.npy", mmap_mode="r")
        y_test = np.load(data_dir / "y_test.npy", mmap_mode="r")

        num_classes = y_train.shape[1]

        if self.fink is not None:
            # Take only G, R bands
            X_train = X_train[:, :, 0:3:2]
            X_test = X_test[:, :, 0:3:2]

        log.info(f"{X_train.shape, y_train.shape}")

        num_samples, timesteps, num_features = X_train.shape

        BATCH_SIZE = find_optimal_batch_size(num_samples)
        log.info(f"BATCH_SIZE:{BATCH_SIZE}")

        input_shape = (BATCH_SIZE, timesteps, num_features)
        log.info(f"input_shape:{input_shape}")

        drop_remainder = False

        def get_compiled_model_and_data(loss, drop_remainder):
            if self.redshift is not None:
                hyper_results_file = f"{self.architecture}/opt/runs/{self.dataset}/results_with_z.json"
                input_shapes = [input_shape, (BATCH_SIZE, Z_train.shape[1])]

                train_ds = (
                    lazy_load_plasticc_wZ(X_train, Z_train, y_train)
                    .shuffle(1000, seed=RANDOM_SEED)
                    .batch(BATCH_SIZE, drop_remainder=drop_remainder)
                    .prefetch(tf.data.AUTOTUNE)
                    .cache()
                )
                test_ds = (
                    lazy_load_plasticc_wZ(X_test, Z_test, y_test)
                    .batch(BATCH_SIZE, drop_remainder=drop_remainder)
                    .prefetch(tf.data.AUTOTUNE)
                    .cache()
                )
            else:
                hyper_results_file = f"{self.architecture}/opt/runs/{self.dataset}/results.json"
                input_shapes = input_shape

                train_ds = (
                    lazy_load_plasticc_noZ(X_train, y_train)
                    .shuffle(1000, seed=RANDOM_SEED)
                    .batch(BATCH_SIZE, drop_remainder=drop_remainder)
                    .prefetch(tf.data.AUTOTUNE)
                    .cache()
                )
                test_ds = (
                    lazy_load_plasticc_noZ(X_test, y_test)
                    .batch(BATCH_SIZE, drop_remainder=drop_remainder)
                    .prefetch(tf.data.AUTOTUNE)
                    .cache()
                )

            # Load hyperparameters
            with open(hyper_results_file, "r") as f:
                hyper_results = json.load(f)

            # Get best hyperparameters
            best_trial = min(hyper_results, key=lambda x: x["value"])
            hyperparameters = best_trial["hyperparameters"]

            # Build model
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=input_shapes),
                tf.keras.layers.LSTM(hyperparameters["units"], return_sequences=True),
                tf.keras.layers.Dropout(hyperparameters["dropout"]),
                tf.keras.layers.LSTM(hyperparameters["units"]),
                tf.keras.layers.Dropout(hyperparameters["dropout"]),
                tf.keras.layers.Dense(num_classes, activation="softmax")
            ])

            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparameters["learning_rate"]),
                loss=loss,
                metrics=["accuracy"]
            )

            return model, train_ds, test_ds, best_trial, hyper_results_file

        # Set up distributed training if multiple GPUs available
        if len(tf.config.list_physical_devices("GPU")) > 1:
            strategy = tf.distribute.MirroredStrategy()
            log.info("Number of devices: {}".format(strategy.num_replicas_in_sync))
            BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
            VALIDATION_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync

            with strategy.scope():
                loss = DistributedWeightedLogLoss(
                    reduction=tf.keras.losses.Reduction.AUTO,
                )
                model, train_ds, test_ds, event, hyper_results_file = get_compiled_model_and_data(loss, drop_remainder)
        else:
            loss = WeightedLogLoss()
            model, train_ds, test_ds, event, hyper_results_file = get_compiled_model_and_data(loss, drop_remainder)

        # Set up callbacks
        callbacks = [
            CSVLogger(csv_logger_file),
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                checkpoint_path,
                monitor="val_loss",
                save_best_only=True
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            SGEBreakoutCallback()
        ]

        # Train model
        history = model.fit(
            train_ds,
            validation_data=test_ds,
            epochs=100,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate model
        log.info(f"PERCENT OF RAM USED: {psutil.virtual_memory().percent}")
        log.info(f"RAM USED: {psutil.virtual_memory().active / (1024*1024*1024)}")

        log.info(f"LL-BATCHED-32 Model Evaluate: {model.evaluate(test_ds, verbose=0)[0]}")
        log.info(f"LL-BATCHED-OP Model Evaluate: {model.evaluate(test_ds, verbose=0, batch_size=VALIDATION_BATCH_SIZE)[0]}")

        if drop_remainder:
            ind = np.array([x for x in range((y_test.shape[0] // BATCH_SIZE) * BATCH_SIZE)])
            y_test = np.take(y_test, ind, axis=0)

        y_preds = model.predict(test_ds)
        log.info(f"{y_preds.shape}, {type(y_preds)}")

        WLOSS = loss(y_test, y_preds).numpy()
        log.info(f"LL-Test Model Predictions: {WLOSS:.8f}")
        if "pytest" in sys.modules:
            return WLOSS

        # Save model
        LABEL = "wZ-" + LABEL if self.redshift else "noZ-" + LABEL
        LABEL = "GR-" + LABEL if self.fink else "UGRIZY-" + LABEL
        LABEL += f"-LL{WLOSS:.3f}"

        if platform.system() != "Darwin":
            model.save(f"{self.architecture}/models/{self.dataset}/model-{LABEL}")
            model.save_weights(f"{self.architecture}/models/{self.dataset}/weights/weights-{LABEL}")

        # Evaluate on test set
        if X_test.shape[0] < 10000:
            batch_size = X_test.shape[0]
        else:
            batch_size = (
                int(VALIDATION_BATCH_SIZE / strategy.num_replicas_in_sync)
                if len(tf.config.list_physical_devices("GPU")) > 1
                else VALIDATION_BATCH_SIZE
            )
            log.info(f"EVALUATE VALIDATION_BATCH_SIZE : {batch_size}")

        # Record metrics
        event["hypername"] = event["name"]
        event["name"] = f"{LABEL}"
        event["z-redshift"] = self.redshift
        event["avocado"] = self.avocado
        event["testset"] = self.testset
        event["fink"] = self.fink
        event["num_classes"] = num_classes
        event["model_evaluate_on_test_acc"] = model.evaluate(test_ds, verbose=0, batch_size=batch_size)[1]
        event["model_evaluate_on_test_loss"] = model.evaluate(test_ds, verbose=0, batch_size=batch_size)[0]
        event["model_prediction_on_test"] = loss(y_test, y_preds).numpy()

        y_test = np.argmax(y_test, axis=1)
        y_preds = np.argmax(y_preds, axis=1)

        event["model_predict_precision_score"] = precision_score(y_test, y_preds, average="macro")
        event["model_predict_recall_score"] = recall_score(y_test, y_preds, average="macro")

        print("  Params: ")
        for key, value in history.history.items():
            print("    {}: {}".format(key, value))
            event["{}".format(key)] = value

        learning_rate = event["lr"]
        del event["lr"]

        return event

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--redshift", action="store_true")
    parser.add_argument("--fink", action="store_true")
    parser.add_argument("--avocado", action="store_true")
    parser.add_argument("--testset", action="store_true")
    args = parser.parse_args()

    training = Training(
        architecture=args.architecture,
        dataset=args.dataset,
        redshift=args.redshift,
        fink=args.fink,
        avocado=args.avocado,
        testset=args.testset,
    )
    training()
