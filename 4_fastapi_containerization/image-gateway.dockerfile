FROM python:3.8.10-slim

# install pipenv
RUN pip install pipenv

# create and cd into app directory
WORKDIR /app

# copy dependencies from pipfile into app
COPY ["Pipfile", "Pipfile.lock", "./"]

# install without creating virtual environment and without changing pipfile
RUN pipenv install --system --deploy

# copy code into app
COPY ["emotion_recognition_prediction_service", "./prediction_service"]

# expose port 9696 to host machine
EXPOSE 9696

# execute this command when the dockerfile is run i.e run our predict service
ENTRYPOINT ["uvicorn", "--port", "9696", "--host","0.0.0.0", "prediction_service.api:app"]