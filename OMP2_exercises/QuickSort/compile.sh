#!/bin/bash
DIR=`dirname $0`

g++-8 -std=c++11 -fopenmp -I"$DIR"/include "$DIR"/QuickSort.cpp -o quicksort
./quicksort
