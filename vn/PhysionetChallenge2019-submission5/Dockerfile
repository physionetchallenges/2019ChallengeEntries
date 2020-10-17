FROM python:3.5.2

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER bytak.lee@gmail.com

RUN mkdir /physionet2019
COPY ./ /physionet2019
WORKDIR /physionet2019
RUN pip install -r requirements.txt


