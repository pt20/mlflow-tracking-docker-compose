FROM python:3.7.8

RUN pip install mlflow psycopg2-binary boto3

ADD . /app
WORKDIR /app

COPY wait-for-it.sh wait-for-it.sh 
RUN chmod +x wait-for-it.sh

EXPOSE 5000