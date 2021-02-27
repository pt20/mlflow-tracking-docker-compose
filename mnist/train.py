import matplotlib.pyplot as plt
import os
import mlflow
import mlflow.tensorflow

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

import tensorflow_datasets as tfds


def load_mnist_dataset():
    (ds_train, ds_test), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    return (ds_train, ds_test), ds_info


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label


def train_pipeline(ds_train, ds_info):
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train


def test_pipeline(ds_test):
    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_test


def create_model():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return model


def fit_model(model, ds_train, ds_val):
    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_val,
    )


# Enable auto-logging to MLflow to capture TensorBoard metrics.
mlflow.tensorflow.autolog()
mlflow.set_experiment("mnist")

def main():

    with mlflow.start_run():
        (ds_train, ds_test), ds_info = load_mnist_dataset()

        ds_train = train_pipeline(ds_train, ds_info)
        ds_test = test_pipeline(ds_test)

        model = create_model()
        fit_model(model, ds_train, ds_test)


if __name__ == "__main__":
    main()
