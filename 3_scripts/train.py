import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

data_file_path = "data/clean_fer_2013/"


def get_image_meta(data_path):
    """Retrieve basic metadata (file_path, category)"""

    file_meta = []

    for name in os.listdir(data_path):
        filepath = os.path.join(data_path, name)

        if ".DS_Store" in filepath:
            continue

        for image_file_name in os.listdir(filepath):
            image_filepath = os.path.join(filepath, image_file_name)

            attrs = {
                "category": name,
                "file_path": image_filepath,
            }
            file_meta.append(attrs)

    return file_meta


rotation_layer = tf.keras.layers.RandomRotation(
    0.05
)  # Rotate by up to 5% of 360 degrees i.e. 18 degrees
zoom_layer = tf.keras.layers.RandomZoom(
    height_factor=0.05, width_factor=0.05
)  # Zoom by 5%
shift_layer = tf.keras.layers.RandomTranslation(
    height_factor=0.05, width_factor=0.05
)  # Shift by 5%
flip_layer = tf.keras.layers.RandomFlip("horizontal")


def augment_image(image):
    """Augment the image with a random rotation, zoom, translation and flip"""
    image = rotation_layer(image)
    image = zoom_layer(image)
    image = shift_layer(image)
    image = flip_layer(image)

    return image


def preprocess_input(image_path, should_augment_image=False):
    """Preprocess Input"""
    try:
        image_file = tf.io.read_file(image_path)
    except:
        print(f"Issue loading {image_file}")
        return None

    image = tf.image.decode_jpeg(image_file, channels=1)
    image = tf.image.resize(image, (48, 48))

    if should_augment_image:
        image = augment_image(image)

    image = image / 255.0
    return image


def create_label_encoder(categories, output_mode="one_hot"):
    """Create a mapping from category names to integer indices and perform one hot encoding by default"""
    return tf.keras.layers.StringLookup(
        vocabulary=categories, output_mode=output_mode, num_oov_indices=0
    )


def create_dataset(
    image_paths, labels, label_encoder, should_augment_image=False, batch_size=32
):
    """Load the dataset from file paths, apply preprocessing and group into batches"""
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(
        lambda x, y: (
            preprocess_input(x, should_augment_image=should_augment_image),
            tf.squeeze(label_encoder(y)),
        )
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(buffer_size=1000)  # Shuffle the dataset
    dataset = dataset.prefetch(
        tf.data.experimental.AUTOTUNE
    )  # Prefetch for better performance
    return dataset


def build_model(learning_rate=0.001, dense_layer_units=[128, 64], drop_rate=0):
    input_shape = (48, 48, 1)
    num_categories = 7

    base_model = tf.keras.Sequential(
        [
            keras.layers.InputLayer(shape=input_shape),
            # First Convolutional Layer (32 filters)
            keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            # Second Convolutional Layer (64 filters)
            keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            # Third Convolutional Layer (128 filters)
            keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
        ]
    )

    # Add additional dense layers
    for unit in dense_layer_units:
        base_model.add(keras.layers.Dense(unit, activation="relu"))
        # Add dropout
        if drop_rate > 0:
            base_model.add(keras.layers.Dropout(drop_rate))

    # Final output layer
    base_model.add(keras.layers.Dense(num_categories, activation="softmax"))

    optimizer = keras.optimizers.Adam(learning_rate)

    loss = keras.losses.CategoricalCrossentropy(from_logits=False)

    base_model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    return base_model


def printArgs(my_func):
    def wrapper(*arg, **kwargs):
        print("{}({})".format(my_func.__name__, ",".join([str(arg), str(kwargs)])))
        return my_func(*arg, **kwargs)

    return wrapper


def calculate_class_weights(df_meta, label_encoder_for_weights):
    # Sort categories alphabetically and get counts
    category_counts = (
        df_meta["category"]
        .value_counts()
        .reset_index()
        .sort_values(by="category")
        .reset_index(drop=True)
    )
    # Get the index for each category from the label encoder
    category_counts["encoded_index"] = category_counts["category"].apply(
        lambda x: label_encoder_for_weights(x).numpy()
    )

    # Calculate the weights
    category_counts["weights"] = (
        np.sum(category_counts["count"]) / category_counts["count"]
    )

    # Normalize the weights
    category_counts["normalized_weights"] = category_counts["weights"] / np.sum(
        category_counts["weights"]
    )

    # Map the encoded index to the weight
    class_weight_dict = dict(
        zip(category_counts["encoded_index"], category_counts["normalized_weights"])
    )

    return class_weight_dict


@printArgs
def train_model(
    learning_rate=0.001,
    dense_layer_units=[64, 32],
    drop_rate=0,
    n_epochs=100,
    use_class_weight=False,
    use_data_augmentation=True,
):
    """Train and validate model and save to file

    Parameters
    ---------
    learning_rate: float, The learning rate, (0,1)
    dense_layer_units: list(int), A list of the number of units for each Dense Layer e.g. 128,64 will create a model with the first dense layer having 128 units and the second having 64
    drop_rate: int, a regularization parameter: the percentage of neurons in a layer to temporarily make inactive [0,1)
    n_epochs: int, the number of epochs to train the model
    use_class_weight: boolean, if True then modify the loss function by the class weights
    use_data_augmentation: boolean, if True then augment all the training images
    """

    base_path = Path(__file__).parent.parent

    logger.info("Prepare the training and validation datasets")

    train_data_folder = base_path / data_file_path / "train/"
    val_data_folder = base_path / data_file_path / "validation/"

    meta_train = get_image_meta(train_data_folder)
    df_meta_train = pd.DataFrame(meta_train)
    train_labels = df_meta_train["category"].values
    train_image_paths = df_meta_train["file_path"].values

    meta_val = get_image_meta(val_data_folder)
    df_meta_val = pd.DataFrame(meta_val)
    val_labels = df_meta_val["category"].values
    val_image_paths = df_meta_val["file_path"].values

    categories = list(df_meta_train["category"].unique())
    label_encoder = create_label_encoder(categories, output_mode="one_hot")
    label_encoder_for_weights = create_label_encoder(categories, output_mode="int")
    logger.info("Calculated the categories: %s", str(categories))

    class_weights = calculate_class_weights(
        df_meta=df_meta_train, label_encoder_for_weights=label_encoder_for_weights
    )
    logger.info("Calculated the class weight dictionary: %s", str(class_weights))

    logger.info(
        "Creating a training dataset where use_data_augmentation=%s",
        use_data_augmentation,
    )
    train_dataset = create_dataset(
        train_image_paths,
        train_labels,
        label_encoder,
        should_augment_image=use_data_augmentation,
        batch_size=32,
    )
    val_dataset = create_dataset(
        val_image_paths,
        val_labels,
        label_encoder,
        should_augment_image=False,
        batch_size=32,
    )

    logger.info(
        "Building a model with learning_rate=%f, dense_layer_units=%s, drop_rate=%d",
        learning_rate,
        dense_layer_units,
        drop_rate,
    )
    model = build_model(
        learning_rate=learning_rate,
        dense_layer_units=dense_layer_units,
        drop_rate=drop_rate,
    )

    summary = model.summary()
    logger.info(summary)

    output_file_path = base_path / "data/er_final_{epoch:02d}_{val_accuracy:.3f}.keras"

    logger.info("Trained models will be saved as: %s", output_file_path)
    checkpoint = keras.callbacks.ModelCheckpoint(
        output_file_path, save_best_only=True, monitor="val_accuracy", mode="max"
    )

    logger.info("Training the model for %d epochs", n_epochs)

    history = None

    if use_class_weight:
        history = model.fit(
            train_dataset,
            epochs=n_epochs,
            validation_data=val_dataset,
            callbacks=[checkpoint],
            class_weight=class_weights,
        )
    else:
        history = model.fit(
            train_dataset,
            epochs=n_epochs,
            validation_data=val_dataset,
            callbacks=[checkpoint],
        )

    best_accuracy = 0
    best_epoch = 0
    for idx, acc in enumerate(history.history["val_accuracy"]):
        if acc > best_accuracy:
            best_accuracy = acc
            best_epoch = idx

    logger.info(
        "The best model was found after %d epochs. It has a validation accuracy of %.2f",
        best_epoch,
        best_accuracy,
    )


if __name__ == "__main__":
    # Default Parameters
    learning_rate_final = 0.0001
    dense_layer_units_final = [128, 64]
    drop_rate_final = 0.0
    n_epochs_final = 100
    use_class_weight_final = False
    use_data_augmentation_final = True

    def list_of_ints(arg):
        list_result = list(map(int, arg.split(",")))
        if all(r == 0 for r in list_result):
            return []
        return list_result

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learning_rate",
        help="The learning rate",
        default=learning_rate_final,
        type=float,
    )

    parser.add_argument(
        "--dense_layer_units",
        help="A list of the number of units for each Dense Layer e.g. 128,64 will create a model with the first dense layer having 128 units and the second having 64",
        default=dense_layer_units_final,
        type=list_of_ints,
    )
    parser.add_argument(
        "--drop_rate",
        help="A regularization parameter: the percentage of neurons in a layer to temporarily make inactive. It should be a float between [0,1)",
        default=drop_rate_final,
        type=float,
    )

    parser.add_argument(
        "--n_epochs",
        help="The number of epochs to train the model",
        default=n_epochs_final,
        type=float,
    )

    parser.add_argument(
        "--use_class_weight",
        help="If True then modify the loss function by the class weights",
        default=use_class_weight_final,
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--use_data_augmentation",
        help="If True then augment all the training images",
        default=use_data_augmentation_final,
        action=argparse.BooleanOptionalAction,
    )

    args = parser.parse_args()

    train_model(
        learning_rate=args.learning_rate,
        dense_layer_units=args.dense_layer_units,
        drop_rate=args.drop_rate,
        n_epochs=args.n_epochs,
        use_class_weight=args.use_class_weight,
        use_data_augmentation=args.use_data_augmentation,
    )
