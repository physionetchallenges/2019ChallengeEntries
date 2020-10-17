sudo docker build -t testimage .

sudo docker run -it -v /home/achuth/sepsis_sub/validate:/data testimage:latest bash
