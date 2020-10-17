# Instructions

The Python script `get_sepsis_score.py` makes predictions on the input data.

To run this script, run

        python get_sepsis_score.py input.psv output.psv

which takes a text file `input.psv` as input and returns a text file `output.psv` as output.

The input files are provided in training database available on the PhysioNet website, and the format for the output files is described on the PhysioNet website.




--------
How to setup Docker 


Build an image enviroment from Dockerfile at current folder

$ docker build -t my-python-app .

Run in a container 

$ docker run my-python-app python get_sepsis_score.py 'p1.psv' 'p1out.psv'

$ docker run my-python-app python driver.py test out

Copy the container output to local system

$ cp {container name}:/physionet2019/p1out.psv p1out.psv


