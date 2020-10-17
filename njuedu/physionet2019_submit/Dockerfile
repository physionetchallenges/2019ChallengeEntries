FROM python:3.7.3-slim

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER yuq329@outlook.com

RUN mkdir /physionet2019
COPY ./ /physionet2019
WORKDIR /physionet2019
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt
