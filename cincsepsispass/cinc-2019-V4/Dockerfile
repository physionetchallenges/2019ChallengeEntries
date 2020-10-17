FROM python:3.6

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER fms@sample.com
## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet2019
COPY ./ /physionet2019
WORKDIR /physionet2019

## Install your dependencies here using apt-get etc.
#RUN apt-get update && \
 #   apt-get install -y python-numpy \
     ##                  python-scipy \
   #                    ipython \                       
      #                 python-pandas \
      #                 python-sympy \
      #                 python-nose \
  #  && apt-get clean \
 #   && apt-get autoclean 

## Do not edit if you have a requirements.txt
RUN pip install -r requirements.txt
