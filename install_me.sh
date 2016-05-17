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
echo "PATH_TO_CRF_TEST='$CRF_PATH/crf_test'" > path_crf.py
echo 


#Install SVM_LIGHT
mkdir svm_light
cd svm_light
wget http://download.joachims.org/svm_light/current/svm_light.tar.gz
gunzip -c svm_light.tar.gz | tar xvf -
make
rm svm_light.tar.gz
cd ..



##Download the models
echo Downloading the trained models, you will be asked for the password during the process
wget --user=cltl --ask-password kyoto.let.vu.nl/~izquierdo/models_opinion_miner_deluxePP.tgz
tar xvzf models_opinion_miner_deluxePP.tgz
rm models_opinion_miner_deluxePP.tgz


wget http://kyoto.let.vu.nl/~izquierdo/public/polarity_models.tgz
tar xvzf polarity_models.tgz
rm polarity_models.tgz

echo All Done

