#!/bin/bash
DIR=`dirname $0`

mpicxx -std=c++14 "$DIR"/ring.cpp -o ring
mpirun -n 4 ./ring
