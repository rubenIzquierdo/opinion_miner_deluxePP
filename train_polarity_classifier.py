#!/usr/bin/env python

from __future__ import print_function
from __future__ import print_function
import argparse
from polarity_classifier import PolarityClassifier




if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description='Train a polarity (positive/negative) classifier for opinions',version='1.0')
    argument_parser.add_argument('-i', dest='inputfile', required=True, help='Input file with a list of paths to KAF/NAF files (one per line)')
    argument_parser.add_argument('-o', dest='output_folder', required=True, help='Folder to store the models')

    args = argument_parser.parse_args()
    
    
    #Load list of files 
    training_files = []
    fd = open(args.inputfile,'r')
    for line in fd:
        if line[0]!='#':
            training_files.append(line.strip())
    fd.close()
    
    print('Total training files: %d' % len(training_files))
    
    my_polarity_classifier = PolarityClassifier('nl')
    my_polarity_classifier.train(training_files, args.output_folder)
