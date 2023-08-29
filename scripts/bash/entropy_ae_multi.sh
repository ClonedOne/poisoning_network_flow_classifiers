#!/bin/bash

# This script is used to run multiple instances of the mimicry attack

python scripts/poison_ctu_ae.py --n_features 8 --seed 42 --fstrat entropy --p_frac 0.005 --subscenario 1;
python scripts/poison_ctu_ae.py --n_features 8 --seed 42 --fstrat entropy --p_frac 0.01 --subscenario 1;
python scripts/poison_ctu_ae.py --n_features 8 --seed 42 --fstrat entropy --p_frac 0.02 --subscenario 1;
python scripts/poison_ctu_ae.py --n_features 8 --seed 42 --fstrat entropy --p_frac 0.04 --subscenario 1;
python scripts/poison_ctu_ae.py --n_features 8 --seed 42 --fstrat entropy --p_frac 0.05 --subscenario 1;
python scripts/poison_ctu_ae.py --n_features 8 --seed 42 --fstrat entropy --p_frac 0.1 --subscenario 1;

python scripts/poison_ctu_ae.py --n_features 8 --seed 23 --fstrat entropy --p_frac 0.005 --subscenario 1;
python scripts/poison_ctu_ae.py --n_features 8 --seed 23 --fstrat entropy --p_frac 0.01 --subscenario 1;
python scripts/poison_ctu_ae.py --n_features 8 --seed 23 --fstrat entropy --p_frac 0.02 --subscenario 1;
python scripts/poison_ctu_ae.py --n_features 8 --seed 23 --fstrat entropy --p_frac 0.04 --subscenario 1;
python scripts/poison_ctu_ae.py --n_features 8 --seed 23 --fstrat entropy --p_frac 0.05 --subscenario 1;
python scripts/poison_ctu_ae.py --n_features 8 --seed 23 --fstrat entropy --p_frac 0.1 --subscenario 1;

python scripts/poison_ctu_ae.py --n_features 8 --seed 2022 --fstrat entropy --p_frac 0.005 --subscenario 1;
python scripts/poison_ctu_ae.py --n_features 8 --seed 2022 --fstrat entropy --p_frac 0.01 --subscenario 1;
python scripts/poison_ctu_ae.py --n_features 8 --seed 2022 --fstrat entropy --p_frac 0.02 --subscenario 1;
python scripts/poison_ctu_ae.py --n_features 8 --seed 2022 --fstrat entropy --p_frac 0.04 --subscenario 1;
python scripts/poison_ctu_ae.py --n_features 8 --seed 2022 --fstrat entropy --p_frac 0.05 --subscenario 1;
python scripts/poison_ctu_ae.py --n_features 8 --seed 2022 --fstrat entropy --p_frac 0.1 --subscenario 1;

python scripts/poison_ctu_ae.py --n_features 8 --seed 0 --fstrat entropy --p_frac 0.005 --subscenario 1;
python scripts/poison_ctu_ae.py --n_features 8 --seed 0 --fstrat entropy --p_frac 0.01 --subscenario 1;
python scripts/poison_ctu_ae.py --n_features 8 --seed 0 --fstrat entropy --p_frac 0.02 --subscenario 1;
python scripts/poison_ctu_ae.py --n_features 8 --seed 0 --fstrat entropy --p_frac 0.04 --subscenario 1;
python scripts/poison_ctu_ae.py --n_features 8 --seed 0 --fstrat entropy --p_frac 0.05 --subscenario 1;
python scripts/poison_ctu_ae.py --n_features 8 --seed 0 --fstrat entropy --p_frac 0.1 --subscenario 1;

python scripts/poison_ctu_ae.py --n_features 8 --seed 1 --fstrat entropy --p_frac 0.005 --subscenario 1;
python scripts/poison_ctu_ae.py --n_features 8 --seed 1 --fstrat entropy --p_frac 0.01 --subscenario 1;
python scripts/poison_ctu_ae.py --n_features 8 --seed 1 --fstrat entropy --p_frac 0.02 --subscenario 1;
python scripts/poison_ctu_ae.py --n_features 8 --seed 1 --fstrat entropy --p_frac 0.04 --subscenario 1;
python scripts/poison_ctu_ae.py --n_features 8 --seed 1 --fstrat entropy --p_frac 0.05 --subscenario 1;
python scripts/poison_ctu_ae.py --n_features 8 --seed 1 --fstrat entropy --p_frac 0.1 --subscenario 1;
