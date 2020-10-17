FROM ubuntu:latest

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER akrammohd@gmail.com; amoham18@uthsc.edu

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.6 \
	python3-setuptools \
    python3-pip \
	python3-dev \
	python3-scipy \ 
	python3-numpy \
	python3-pandas \
	python3-sklearn \
	python3-joblib \
    vim \
    && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

# set python 3 as the default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1

RUN pip3 install cloudpickle==0.5.6
RUN pip3 install scikit-learn==0.19.1
RUN pip3 install keras
RUN pip3 install tensorflow

## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet2019
COPY ./ /physionet2019
WORKDIR /physionet2019

## Install your dependencies here using apt-get etc.

## Do not edit if you have a requirements.txt
#RUN pip install -r requirements.txt

