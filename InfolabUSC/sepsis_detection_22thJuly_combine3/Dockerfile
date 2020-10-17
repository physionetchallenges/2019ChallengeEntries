FROM python:2.7-stretch

 

## The MAINTAINER instruction sets the Author field of the generated images

MAINTAINER author@sample.com

## DO NOT EDIT THESE 3 lines

RUN mkdir /physionet2019

COPY ./ /physionet2019

WORKDIR /physionet2019

 

## Install your dependencies here using apt-get etc.

RUN apt-get -y update

RUN apt-get install -y build-essential

 

## Do not edit if you have a requirements.txt

RUN pip install -r requirements.txt
