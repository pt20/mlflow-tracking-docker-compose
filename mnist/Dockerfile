FROM python:3.7.8

RUN pip install --upgrade pip
RUN pip install boto3

ADD . /app
WORKDIR /app

RUN pip install -r requirements.txt
