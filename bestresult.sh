#!/bin/bash

statistic=${1:-PRF}
beta=${2:-0.5}
results_file=${3:-vw/results}
best_file=${4:-vw/best_params}
best_statistic=0
new_statistic=0
while read line; do
    if [[ $line == weight* ]]; then
        # weight line starts with "weight: "
        weight=$(echo $line | grep -o -P '(?<=weight: )[0-9]+';)
    fi
    if [[ $line == --* ]]; then
        # params line starts with "--passes"
        params=$line
    fi
    if [[ $statistic == F ]]; then
        if [[ $line == PRE* ]]; then
            precision=$(echo $line | grep -o -P "(?<=PRE )[0-9]+([.][0-9]+)";)
        fi
        # recall always comes after precision
        if [[ $line == REC* ]]; then
            recall=$(echo $line | grep -o -P "(?<=REC )[0-9]+([.][0-9]+)";)
            if [[ $precision != 0 ]]; then
                # F score
                new_statistic=$(bc -l <<< "(1+$beta^2) * $precision * $recall / (($beta^2 * $precision) + $recall)")
            fi
        fi
    else
        if [[ "$line" == $statistic* ]]; then
            new_statistic=$(echo $line | grep -o -P "(?<=$statistic )[0-9]+([.][0-9]+)";)
        fi
    fi
    if [[ $(echo $new_statistic'>'$best_statistic | bc -l) == 1 ]]; then
        best_statistic=$new_statistic
        best_weight=$weight
        best_params=$params
        # these variables do not persist outside the loop?
        echo $statistic: $best_statistic > $best_file
        echo weight: $best_weight >> $best_file
        echo $best_params >> $best_file
    fi
done < $results_file


