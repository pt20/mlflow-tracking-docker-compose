FROM python:3.7.8

RUN pip install --upgrade pip

RUN pip install mlflow psycopg2-binary scikit-learn boto3

ADD . /app
WORKDIR /app
