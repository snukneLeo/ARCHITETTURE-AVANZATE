#!/bin/bash
DIR=`dirname $0`

g++-8 -std=c++11 -fopenmp -I"$DIR"/include ${DIR}/RC4.cpp ${DIR}/RC4_p.cpp ${DIR}/main.cpp -o rc4
./rc4
