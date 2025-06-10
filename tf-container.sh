#!/bin/bash
#to run it: ./tf-container.sh

docker build -t sat-iq .
#docker run --gpus all -it --rm -u $(id -u):$(id -g) -v $(pwd):/code -v $(pwd)/../data:/data sat-iq
docker run -it --rm -u $(id -u):$(id -g) -v $(pwd):/code -v $(pwd)/../data:/data sat-iq
