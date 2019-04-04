#!/bin/bash
DIR=`dirname $0`

g++-8 -std=c++11 -fopenmp "$DIR"/Factorial.cpp -I"$DIR"/include -o factorial
./factorial
