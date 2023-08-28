#!/bin/bash

for modeltype in GradientBoosting FFNN 
do 
    for i in {0..4}
    do
        python scripts/poison_ctu.py --model $modeltype --n_features 8 --seed $i --fstrat random --subscenario 1;
    done
done
