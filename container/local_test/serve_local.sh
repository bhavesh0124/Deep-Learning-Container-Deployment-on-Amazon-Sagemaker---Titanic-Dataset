#!/bin/sh

image=$1

sudo docker run -v "$(pwd)/test_dir:/opt/ml" -p 8080:8080 --rm ${image} serve
