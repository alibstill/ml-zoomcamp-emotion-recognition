# Containerization and Local Deployment

The Emotion Recognition Prediction Service is a simple [FastAPI](https://fastapi.tiangolo.com/) application. It has a single endpoint (`predict-emotion`) where a file can be uploaded. 

When an image file is received, the FastAPI application makes a request to a [`tensorflow/serving`](https://hub.docker.com/r/tensorflow/serving) application which houses the emotion recognition model. The model then predicts the emotion and sends the prediction to the emotion recognition service. The emotion recognition service parses the prediction and sends the result back to the user.

## Run the app locally

The easiest way to run the service locally, given that there are 2 apps, is to use `docker-compose`:

```bash
docker-compose up
```

## Making a request
After getting the app running, navigate to `http://localhost:9696/docs` to view the API documentation and experiment with the API directly.

Please make sure you upload a `.jpg`.

### Validation and Error handling

Please note that this is currently a very simple app with little error handling or validation. It is up to the user to choose a suitable file format and file.





