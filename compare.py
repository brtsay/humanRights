#!/usr/bin/env python3

import argparse

description = 'This script takes a file with raw predictions and a labeled set in VW format and outputs a file that contains the misclassified posts. The format is [pred] | [true label] | [rest of post]'
parser = argparse.ArgumentParser(description=description)

parser.add_argument('-r', '--rawpred', help='File with the raw predictions')
parser.add_argument('-t', '--test', help='Labeled test set file')
parser.add_argument('-o', '--output',help='Where to save mislabeled tweets')

args = parser.parse_args()

with open(args.rawpred, 'r') as rawpred_file, open(args.test, 'r') as test_file, open(args.output, 'w') as output_file:
    for predline, testline in zip(rawpred_file, test_file):
        if predline[0] == '-':  # check negative
            prediction = "-1"
        else:
            prediction = "+1"
        # VW files begin with something like "+1 | "
        truelabel = testline.split('|')[0].strip()
        if prediction != truelabel:
            output_file.write(prediction+' | '+testline+'\n')

