#!/bin/bash
DIR=`dirname $0`

g++ -std=c++11 -O0 -fopenmp -I"$DIR"/include "$DIR"/ProducerConsumer.cpp -o producerconsumer
./producerconsumer
