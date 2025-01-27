FROM tensorflow/serving:2.7.0

COPY ./emotion_recognition_model/ /models/emotion_recognition_model/1

ENV MODEL_NAME="emotion_recognition_model"