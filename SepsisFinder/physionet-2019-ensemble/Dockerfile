FROM continuumio/anaconda3



## The MAINTAINER instruction sets the Author field of the generated images

MAINTAINER aut...@sample.com

## DO NOT EDIT THESE 3 lines

RUN mkdir /physionet2019

COPY ./ /physionet2019

WORKDIR /physionet2019



## Install your dependencies here using apt-get etc.


ENV PATH /opt/conda/bin:$PATH
RUN conda config --set ssl_verify no
RUN conda create --name physionet
RUN activate physionet
RUN conda install pandas numpy scikit-learn scipy
RUN conda install pytorch-cpu torchvision-cpu -c pytorch
RUN conda install -c conda-forge keras
RUN conda install -c anaconda joblib
