#!/bin/bash

# https://github.com/hal3/vwnlp/blob/master/GettingStarted.ipynb
# https://github.com/hal3/vwnlp/blob/master/GettingTheMost.ipynb

# a makes indexed arrays
# declare -a files=("train" "test")
files=("train" "test")

# need to change label format from __label__LABEL to LABEL |
# would need to replace colons and pipes, but all punctuation already removed
# files for fasttext already removed URLs, cases, and punctuation
# also need to shuffle because all the positive examples in a row
# vw does online learning
# for i in "${files[@]}"
# do
#     # shuf is to upweight postive examples
#     cat fast$i.txt | awk '{sub("__label__hr", "+1 '$(shuf -i 1-10 -n 1)'|w"); sub("__label__nonhr", "-1 |w"); print $0" |l len:" length($0)}'| shuf > ./vw/vw$i.txt
#     head vw/vw$i.txt
# done

# training data
# upweight positive examples
weight=$(shuf -i 1-10 -n 1)
cat fasttrain.txt | awk '{sub("__label__hr", "+1 '$weight'|w"); sub("__label__nonhr", "-1 |w"); print $0" |l len:" length($0)}'| shuf > vw/vwtrain.txt
# validation data
cat fasttest.txt | awk '{sub("__label__hr", "+1 |w"); sub("__label__nonhr", "-1 |w"); print $0" |l len:" length($0)}'| shuf > vw/vwtest.txt
echo weight: $weight >> vw/results

# train vw model
# passes refer to the number of passes over the data
# -c tells vw to use a cache file
# (useful when doing multiple passes, since lots of time on data i/o)
# -k says kill the old cache file
# -f says to make a file for the model
# -b how many "bits" to use, more bits -> more features
# 18 is default, means 2^18 entries in feature vector
# affix is for pre-/suf-fixes, tells vw how many characters forwards/backwards to use as features
# spelling _ : where vw keeps track of word forms eg BLah -> AAaa

n_passes=$(shuf -i 5-100 -n 1)
ngrams=$(shuf -i 1-5 -n 1)
n_skips=$(shuf -i 0-3 -n 1)
loss_type_array=("logistic" "hinge")
loss_type=${loss_type_array[$RANDOM % ${#loss_type_array[@]}]}
params="--passes $n_passes --loss_function $loss_type --ngram w$ngrams --skips w$n_skips"

if (($RANDOM < 32767/2)); then
    # quadratic on w[ords] and l[ength] namespaces
    params=$params" -q wl"
fi

# whether to include regularization
gen_reg_strength() {
    power=$(shuf -i 4-6 -n 1)
    reg_strength=$(bc -l <<< "10^-$power")
    echo $reg_strength
}

if (($RANDOM < 32767/2)); then
    if (($RANDOM < 32767/2)); then
        params=$params" --l1 $(gen_reg_strength)"
    fi
    if (($RANDOM < 32767/2)); then
        params=$params" --l2 $(gen_reg_strength)"
    fi
fi

# mlp
# if (($RANDOM < 32767/2)); then
#     params=$params" --nn $(shuf -i 1-20 -n 1) --dropout"
#     if (($RANDOM < 32767/2)); then
#         # include connections from input layer to output layer
#         params=$params" --inpass"
#     fi
# fi
echo $params

vw --binary vw/vwtrain.txt -c -k -f vw/vw.model -b 24 $params

# try out on test set
# vw --binary -t -i vw/vw.model -p vw/vwtest_pred.txt vw/vwtest.txt
# get raw scores
vw --binary -t -i vw/vw.model -r vw/vwtest_rawpred.txt vw/vwtest.txt

# calculate precision
cut -d' ' -f1 vw/vwtest.txt | paste - vw/vwtest_rawpred.txt | perf.linux/perf -t 0 -PRE -REC -PRF -PRB -ACC >> vw/results


# process unlabeled text
# cat first200k.csv | awk -F "\"*,\"*" '{print $12}' | sed 's/http[^ ]*//g' | tr -d '[:punct:]' | awk '{print "|w "$0}' | awk -f preproc.awk > vw/vwfirst200k.txt

# predict unlabeled text
vw --binary -t -i vw/vw.model -p vw/vwunlabeled_pred.txt vw/vwfirst200k.txt

# paste the labels with the txt to make looking through easier
paste vw/vwunlabeled_pred.txt vw/vwfirst200k.txt > vw/vwlabeled200k.txt

# look at some of the random tweets labeled as human rights related
grep  '^1.*' vw/vwlabeled200k.txt | shuf -n 15
