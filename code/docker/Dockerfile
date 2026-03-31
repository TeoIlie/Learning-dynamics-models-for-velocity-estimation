FROM nvidia/cuda:11.6.2-base-ubuntu20.04 AS base

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3-pip python-is-python3 git vim python3-tk

COPY requirements.txt .
RUN pip install -U pip  \
    && pip install -r requirements.txt

ENV PYTHONPATH=/learning_through_kalman_filter
