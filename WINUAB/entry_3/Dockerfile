FROM python:3.5-slim

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER edwar.macias@uab.cat

RUN mkdir /physionet2019
COPY ./ /physionet2019
WORKDIR /physionet2019
RUN pip install -r requirements.txt
