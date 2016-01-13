#!/bin/bash

#Install the KafNafParserPy
git clone https://github.com/cltl/KafNafParserPy

#Install CRF++
cd crf_lib
tar xvzf CRF++-0.58.tar.gz
cd CRF++-0.58
./configure
make
CRF_PATH=`pwd`
cd ..
cd ..
echo PATH_TO_CRF_TEST="$CRF_PATH/crf_test" > path_crf.py
echo 
echo All Done

