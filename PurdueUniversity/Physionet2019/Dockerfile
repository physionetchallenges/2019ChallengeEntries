FROM python:3.6.7

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER aalthof@purdue.edu
## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet2019
COPY ./ /physionet2019
WORKDIR /physionet2019

## Install your dependencies here using apt-get etc.

## Do not edit if you have a requirements.txt
RUN pip install -r requirements.txt

