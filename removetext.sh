#!/bin/bash

labeled=("train" "valid" "test")

for labeledtype in "${labeled[@]}";
do
    # remove "rt"
    cat fast${labeledtype}.txt | sed -e 's/\<rt\>//g' > fast${labeledtype}_proc.txt
    # remove "thank*"
    sed -i '/thank/d' fast${labeledtype}_proc.txt
    echo Original number of tweets:$(wc -l < fast${labeledtype}.txt)
    echo New number of tweets:$(wc -l < fast${labeledtype}_proc.txt)
done
