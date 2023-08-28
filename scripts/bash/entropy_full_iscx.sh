#!/bin/bash

for modeltype in GradientBoosting FFNN 
do 
    for conntype in "file_video" "chat_video"
    do
        for i in {0..4}
        do
            python scripts/poison_iscx.py --subscenario $conntype --model $modeltype --n_features 8 --seed $i --fstrat entropy;
        done
    done
done
