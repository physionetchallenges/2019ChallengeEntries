## Use

You can create a docker image firstly by running

    docker build -t hzl_challenge:v1 ./

Then create a container by the image

    docker run --name myChallenge1 -v /usr/hzlTensorflow/challenge19:/physionet2019 -dit hzl_challenge:v1 
    
Now the container `myChallenge1` has been created, get into this container and then you can test our model

    docker exec -it myChallenge1 bash
