FROM python:3.7.3-slim

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER soufiane.chami@und.edu
## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet2019
COPY ./ /physionet2019
WORKDIR /physionet2019

## Install your dependencies here using apt-get etc.
RUN apt-get update && apt-get install -y libgomp1 
## Do not edit if you have a requirements.txt
RUN pip install -r requirements.txt

