#!/bin/bash
DIR=`dirname $0`

nvcc -std=c++11 -arch=sm_62 "$DIR"/HelloWorld.cu -o hello_world
