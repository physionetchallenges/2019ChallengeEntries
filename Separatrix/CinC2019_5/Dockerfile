FROM python:3.5.5-stretch

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER morteza.zabihi@gmail.com
## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet2019
COPY ./ /physionet2019
WORKDIR /physionet2019

## Install your dependencies here using apt-get etc.
## Install XGBoost library
##RUN git clone --recursive https://github.com/dmlc/xgboost && \
##    cd xgboost && \
##    make -j4 && \
##    cd python-package; python setup.py install
    
## Do not edit if you have a requirements.txt
RUN pip install -r requirements.txt

