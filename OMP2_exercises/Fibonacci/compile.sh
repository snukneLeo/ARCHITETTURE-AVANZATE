#!/bin/bash
DIR=`dirname $0`

g++-8 -std=c++11 -fopenmp -I"$DIR"/include "$DIR"/Fibonacci.cpp -o fibonacci
./fibonacci

