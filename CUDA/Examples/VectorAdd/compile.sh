#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-7.5/bin:$PATH
DIR=`dirname $0`

nvcc -w -std=c++11 "$DIR"/VectorAdd.cu -I"$DIR"/include -o vector_add
