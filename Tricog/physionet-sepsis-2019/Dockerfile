FROM python:3.7.3-stretch

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER author@sample.com
## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet2019
COPY ./ /physionet2019
WORKDIR /physionet2019
RUN mkdir /physionet2019/models
RUN wget -i downlinks.txt -P  /physionet2019/models
## Install your dependencies here using apt-get etc.
RUN apt-get update
RUN apt-get install apt-utils build-essential python3-pandas python3-scipy| true
## Do not edit if you have a requirements.txt
RUN pip3 install -r requirements.txt

