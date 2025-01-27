from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import io
import grpc

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.core.framework import tensor_pb2, tensor_shape_pb2, types_pb2

import os
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

app = FastAPI(title="Emotion Recognition Prediction Service")

host = os.getenv("TF_SERVING_HOST", "localhost:8500")
timeout = 20.0

channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


categories = ["happy", "sad", "fear", "surprise", "neutral", "angry", "disgust"]


def prepare_response(response):
    preds = response.outputs["output_0"].float_val
    logger.info("Predictions: %s", preds)
    return dict(zip(categories, preds))


def dtypes_as_dtype(dtype):
    if dtype == "float32":
        return types_pb2.DT_FLOAT
    raise Exception("dtype %s is not supported" % dtype)


def make_tensor_proto(data):
    shape = data.shape
    dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=i) for i in shape]
    proto_shape = tensor_shape_pb2.TensorShapeProto(dim=dims)

    proto_dtype = dtypes_as_dtype(data.dtype)

    tensor_proto = tensor_pb2.TensorProto(dtype=proto_dtype, tensor_shape=proto_shape)
    tensor_proto.tensor_content = data.tobytes()

    return tensor_proto


def np_to_protobuf(data):
    if data.dtype != "float32":
        data = data.astype("float32")
    return make_tensor_proto(data)


def build_request(image_array):
    pb_request = predict_pb2.PredictRequest()

    # Provide the model name, "MODEL_NAME",
    pb_request.model_spec.name = "emotion_recognition_model"

    # Provide the signature name
    pb_request.model_spec.signature_name = "serving_default"

    # Specify the name of the input and pass in the image input
    pb_request.inputs["inputs"].CopyFrom(np_to_protobuf(image_array))
    return pb_request


def get_prediction(image_array):
    request = build_request(image_array)
    logger.info("Making request to model service.")
    pb_response = stub.Predict(request, timeout=timeout)
    return prepare_response(pb_response)


def preprocess_image(image):
    """Preprocess image using Pillow"""
    try:
        # Convert the image to grayscale
        image = image.convert("L")
    except Exception as e:
        raise ValueError(f"Invalid image. Error: {e}")

    # Resize the image to 48x48 pixels
    image = image.resize((48, 48))

    # Convert the image to a NumPy array and normalize
    image_array = np.array(image) / 255.0

    # Add batch and channels dimensions to make the shape (1, 48, 48, 1)
    image_array = np.expand_dims(
        image_array, axis=-1
    )  # Add channel dimension (grayscale)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


@app.post("/predict-emotion")
async def upload_image(file: UploadFile = File(...)):
    # Read the image file
    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes))

    logger.info("Received image with size: %s, mode: %s", image.size, image.mode)

    image_array = preprocess_image(image)

    pred_probs = get_prediction(image_array)

    prediction = max(pred_probs, key=pred_probs.get)

    return {"prediction": prediction, "probabilities": pred_probs}
