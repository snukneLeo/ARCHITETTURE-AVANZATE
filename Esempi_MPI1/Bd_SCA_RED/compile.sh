#!/bin/bash
DIR=`dirname $0`

mpicxx -std=c++14 "$DIR"/collective.cpp -o collective
