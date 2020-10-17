FROM r-base:latest

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER hshaider@ualberta.ca

RUN mkdir /physionet2019
COPY ./ /physionet2019
WORKDIR /physionet2019


RUN R -e 'install.packages(c("dplyr", "tidyr", "xgboost"))'
