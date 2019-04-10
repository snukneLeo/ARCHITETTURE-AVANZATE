#!/bin/bash
DIR=`dirname $0`

g++ -std=c++11 -O0 -fopenmp -I"$DIR"/include "$DIR"/QuickSort.cpp -o quicksort
COUNTER=0
while [  $COUNTER -lt 1 ]; do
    ./quicksort
    let COUNTER=COUNTER+1 
done
