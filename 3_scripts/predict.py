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


def preprocess_input(image_path):
    """Preprocess Input"""
    try:
        image_file = tf.io.read_file(image_path)
    except:
        raise ValueError("Issue loading file")

    try:
        # Use channel = 1 to set images to grayscale whether grayscale or RGB
        image = tf.image.decode_jpeg(image_file, channels=1)
    except tf.errors.InvalidArgumentError:
        raise ValueError(
            "Invalid image_path: %s. Only `.jpg` files are acceptable", image_file
        )

    image = tf.image.resize(image, (48, 48))

    image = image / 255.0
    return image


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


def predict(model_file_path, image_file_path):
    """Train and validate model and save to file

    Parameters
    ---------
    model_file_path: The path to the model. Note this should be relative to the root folder
    image_file_path: The path to the image file you want to predict.
    """
    base_path = Path(__file__).parent.parent

    # Get categories
    # train_data_folder = base_path / "data/clean_fer_2013/train/"
    # meta_train = get_image_meta(train_data_folder)
    # df_meta_train = pd.DataFrame(meta_train)
    # categories = list(df_meta_train["category"].unique())
    # category_dict = dict(zip(np.arange(len(categories)), categories))
    # print(categories)
    categories = ["happy", "sad", "fear", "surprise", "neutral", "angry", "disgust"]
    category_dict = dict(zip(np.arange(len(categories)), categories))

    # Load model
    model_filepath = base_path / model_file_path

    logger.info("Loading model from %s", model_file_path)
    model = keras.models.load_model(model_filepath)

    logger.info("Loading image from %s\n. Preparing for prediction.", image_file_path)
    image_path = base_path / image_file_path
    img = preprocess_input(str(image_path))
    img = tf.expand_dims(img, axis=0)

    logger.info("Making prediction")
    pred = model.predict(img)
    predictions = {category_dict[k]: v for k, v in enumerate(pred[0])}
    predicted_class_index = np.argmax(pred)
    predicted_class = categories[predicted_class_index]

    logger.info("Probabilities of classes: %s", str(predictions))
    logger.info("Predicted class: %s", predicted_class)


if __name__ == "__main__":
    model_file_path_final = "2_model_training/er_final_with_aug_94_0.566.keras"
    image_file_path_final = "3_scripts/example_happy.jpg"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file_path",
        help="The path to the trained model",
        default=model_file_path_final,
        type=str,
    )

    parser.add_argument(
        "--image_file_path",
        help="The path to the file of the image you want to predict",
        default=image_file_path_final,
        type=str,
    )

    args = parser.parse_args()

    predict(model_file_path=args.model_file_path, image_file_path=args.image_file_path)
