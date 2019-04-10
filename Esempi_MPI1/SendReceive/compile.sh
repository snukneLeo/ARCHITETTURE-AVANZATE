#!/bin/bash
DIR=`dirname $0`

mpicxx -std=c++14 "$DIR"/SendReceive.cpp -o sendreceive
