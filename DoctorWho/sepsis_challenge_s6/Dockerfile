# FROM python:3.7.3-slim
FROM tensorflow/tensorflow:1.9.0-gpu-py3
#FROM tensorflow/tensorflow:1.10.0-py3

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER michael.moor@bsse.ethz.ch
## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet2019
COPY ./ /physionet2019
WORKDIR /physionet2019

## Install your dependencies here using apt-get etc.

## Do not edit if you have a requirements.txt
RUN pip install -r requirements.txt

