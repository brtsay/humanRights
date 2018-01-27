#!/bin/bash

# https://github.com/hal3/vwnlp/blob/master/GettingStarted.ipynb
# https://github.com/hal3/vwnlp/blob/master/GettingTheMost.ipynb

random_iter=${1:-2}
force_l1=${2:-false}
# whether to supplement the labeled data with unlabeled data
# and treat all the unlabeled as non-human rights
supp_train=${3:-false}
model_name=${4:-vw/vwmodel}
results_file=${5:-vw/vwresults}
fasttrain=${6:-fasttrain.txt}
fastvalid=${7:-fastvalid.txt}
fasttest=${8:-fasttest.txt}


# need to change label format from __label__LABEL to LABEL |
# would need to replace colons and pipes, but all punctuation already removed
# files for fasttext already removed URLs, cases, and punctuation
# also need to shuffle because all the positive examples in a row
# vw does online learning

# generate regularization strength
gen_reg_strength() {
    power=$(shuf -i 4-7 -n 1)
    reg_strength=$(bc -l <<< "10^-$power")
    echo $reg_strength
}

# generate learning rate
gen_lr() {
    minusby=$(shuf -i 0-9 -n 1)
    lr=$(bc -l <<< "1-$minusby/10")
    echo $lr
}


if [[ "supp_train" = true ]]; then
    n_supp_lines=$(wc -l < first200k.csv)
    leftover=$(bc <<< "$n_supp_lines - 50000")
    tail -$leftover vw/vwfirst200k.txt > vw/vwlast150k.txt
fi

# only format validation and test sets
# training data is made in loop
files=("valid" "test")
for data in "${files[@]}";
do
    cat fast${data}.txt | awk '{sub("__label__hr", "+1 |w"); sub("__label__nonhr", "-1 |w"); print $0}'| shuf > vw/${data}tmpW
    cat vw/${data}tmpW | sed 's:.*|w::'| awk '$0="|l len:"length($0)' > vw/${data}tmpL
    paste vw/${data}tmpW vw/${data}tmpL > vw/vw${data}.txt
    rm vw/${data}tmp*
done

for i in in `seq 1 $random_iter`;
do
    # upweight positive examples
    weight=$(shuf -i 1-10 -n 1)
    echo weight: $weight >> $results_file
    # training data
    # this part in loop since weight must be written in data
    cat fasttrain.txt | awk '{sub("__label__hr", "+1 '$weight'|w"); sub("__label__nonhr", "-1 |w"); print $0}' | shuf > vw/traintmpW
    # validation/test data
    cat vw/traintmpW | sed 's:.*|w::'| awk '$0="|l len:"length($0)' > vw/traintmpL
    paste vw/traintmpW vw/traintmpL > vw/vwtrain.txt
    rm vw/traintmp*

    if [[ "$supp_train" = true ]]; then
        head -50000 vw/vwfirst200k.txt | awk '{print("-1 "$0)}' | shuf >> vw/vwsupp.txt
        head -30000 vw/vwsupp.txt > vw/vwtrain_supp.txt
        tail -15000 vw/vwsupp.txt >> vw/vwtest.txt
        cat vw/vwsupp.txt | awk 'NR > 30000 && NR < 35000' >> vw/vwvalid.txt
    fi

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

    n_passes=$(shuf -i 5-20 -n 1)
    ngrams=$(shuf -i 1-3 -n 1)
    n_skips=$(shuf -i 0-3 -n 1)
    loss_type_array=("logistic" "hinge")
    loss_type=${loss_type_array[$RANDOM % ${#loss_type_array[@]}]}
    params="--passes $n_passes --loss_function $loss_type --ngram w$ngrams --skips w$n_skips --learning_rate $(gen_lr)"

    if (($RANDOM < 32767/2)); then
        # quadratic on w[ords] and l[ength] namespaces
        params=$params" -q wl"
    fi

    # whether to include L1 or L2 regularization
    if [ "$force_l1" = false ]; then
        if (($RANDOM < 32767/2)); then
            if (($RANDOM < 32767/2)); then
                params=$params" --l1 $(gen_reg_strength)"
            fi
            if (($RANDOM < 32767/2)); then
                params=$params" --l2 $(gen_reg_strength)"
            fi
        fi
    else
        params=$params" --l1 $(gen_reg_strength)"
    fi
        
    # mlp
    # if (($RANDOM < 32767/2)); then
    #     params=$params" --nn $(shuf -i 1-20 -n 1) --dropout"
    #     if (($RANDOM < 32767/2)); then
    #         # include connections from input layer to output layer
    #         params=$params" --inpass"
    #     fi
    # fi
    echo $params >> $results_file

    vw --binary vw/vwtrain.txt -c -k -f vw/vw.model -b 24 $params

    # try out on validation set
    # vw --binary -t -i vw/vw.model -p vw/vwvalid_pred.txt vw/vwvalid.txt
    # get raw scores
    vw --binary -t -i vw/vw.model -r vw/vwvalid_rawpred.txt vw/vwvalid.txt

    # calculate precision
    cut -d' ' -f1 vw/vwvalid.txt | paste - vw/vwvalid_rawpred.txt | perf.linux/perf -t 0 -PRE -REC -PRF -PRB -ACC >> $results_file

done
