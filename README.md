# Emotion Recognition Prediction

## Scenario
Some people have difficulty recognising emotions on faces. It would be fun to build a wearable piece of technology that could identify emotions of other people while the user was interacting with them in realtime.

This project is a first step in that direction. It aims to build a model that is capable of classifying images of faces into 7 different emotional categories: "happy", "sad", "fear", "surprise", "neutral", "angry", "disgust".

## Data used to build the model

The dataset was downloaded from [kaggle](https://www.kaggle.com/datasets/msambare/fer2013/data) and derived from a [2013 Facial Recognition Kaggle competition](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).

## Navigating this project

This project documents the journey to creating the final prediction model and application.

- [1_eda](1_eda/): exploratory data analysis
- [2_model_training](2_model_training/): I trained, tuned and evaluated 3 different models 
- [3_scripts](3_scripts/): contains scripts to train the final model and get predictions
- [4_fastapi_containerization](4_fastapi_containerization/): contains the fastapi app

## The final app

### Running the final app

Navigate to the [4_fastapi_containerization](4_fastapi_containerization/) folder and run:

```bash
cd 4_fastapi_containerization
docker-compose up -d
```

Once running, navigate to `http://localhost:8060/docs` to view and test out the API.
