#!/bin/bash
DIR=`dirname $0`

g++ -std=c++11 -fopenmp "$DIR"/ProducerConsumer.cpp -o producerconsumer
