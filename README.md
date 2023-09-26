# Poisnet

Code for the paper: Poisoning Network Flow Classifiers, ACSAC 2023

## Description

The code in this repository can be used to reproduce the experiments reported in Sections 5.2-5.5 of the paper. 
We present a new methodology to perform backdoor poisoning attacks against network flow classifiers.
Our approach focuses on the manipulation of aggregated traffic flow features rather than packet-level content, and in particular statistical features on connection metadata extracted with [Zeek](https://zeek.org/).

The codebase is written completely in Python and tested on version 3.8.13 on a machine with Ubuntu server 22.04.1 LTS, 2 Intel Xeon E5-2680 v4, and 128GB RAM.
Note that the experiments were run with an Nvidia Titan X (Pascal) GPU with 12GB of VRAM.
However, due to the relatively small sizes of the models used, the experiments can be run on CPU without major slowdowns.


## Setup

### Installation

We strongly suggest using a Python virtual enviroment to run this project.
```bash
conda create -n poisnet python=3.8.13
conda activate poisnet
```

**Note:** The code depends on `tensorflow==2.11.0`. Please install this dependency before running the code. Depending on your environment the installation process for tensorflow may vary.

Install the other dependencies and the project with:

```bash
pip install -r requirements.txt
pip install -e .
```


### Data

For ease of access, we are providing the pre-processed Zeek `conn.log` files in the form of `.csv` files at the following address: https://drive.google.com/file/d/1Q0R9LZr9-4CachGivhfQac_oLnc57kZo/view?usp=sharing

These directories contain the data already split in the same way used in the paper. 
The compressed archive also contains the trained auto-encoder model for the experiments in Table 4.

After downloading and de-compressing the data, change the paths in `netpois/constants.py` to point to the relevant directories in your machine.
Please ensure that the drive where the data directory is located has ample storage space available.

#### Raw data

Alternatively, the original datasets, and Zeek, can be downloaded from the respective websites:
- CTU-13 dataset: https://www.stratosphereips.org/datasets-ctu13
- CIC ISCXVPN 2016 dataset: https://www.unb.ca/cic/datasets/vpn.html
- CIC IDS 2018 dataset: https://www.unb.ca/cic/datasets/ids-2018.html
- The Zeek Network Security Monitor: https://zeek.org/

Run Zeek on the folders containing the pcap to extract the conn.log files. 
An example of how to run Zeek on all files in a folder is provided in `scripts/bash/zeekit.sh`.

An example function to read the `conn.log` files is provided in `notebooks/data/cic_ids2018.ipynb`.


### Project structure

``` bash
.
├── notebooks                   # Jupyter Notebook files for data analysis and visualizations
│   ├── data                    # Notebooks for the initial data analysis
│   │   ├── cic_ids2018.ipynb
│   │   ├── ctu_13.ipynb
│   │   └── iscx_nonvpn.ipynb
│   └── visualizations          # Notebooks to visualize the results of the experiments and reproduce the plots from the paper
│       ├── ctu13_ae_results.ipynb
│       ├── ctu13_results.ipynb
│       ├── other_results.ipynb
│       ├── poisoning_divergence_benign.ipynb
│       └── saved_files         # JS distances computed by poisoning_divergence_benign.ipynb are saved here
├── poisnet                     # Main code library
│   ├── autoencoder_models.py   # Model definitions 
│   ├── constants.py            # Constant values used throughout the code base
│   ├── ctu_utils.py
│   ├── data_utils.py
│   ├── nn_models.py            # Model definitions 
│   ├── poison                  # Utilities for the poisoning attack
│   │   ├── mimicry_utils.py
│   │   ├── shapbdr_constants.py
│   │   ├── shapbdr_feature_selectors.py
│   │   └── shapbdr_utils.py
│   └── utils.py
├── scripts                     # Executable scripts to run the experiments
│   ├── bash                    # Orchestration scripts to run multiple experiments. Name format is `FeatureSelectionStrategy_TriggerType`
│   │   ├── entropy_ae_multi.sh
│   │   ├── entropy_full.sh
│   │   ├── entropy_full_cicids.sh
│   │   ├── entropy_full_iscx.sh
│   │   ├── entropy_gen.sh
│   │   ├── entropy_red.sh
│   │   ├── gini_full.sh
│   │   ├── random_full.sh
│   │   ├── shap_full.sh
│   │   ├── shap_full_ccids.sh
│   │   ├── shap_full_iscx.sh
│   │   ├── shap_gen.sh
│   │   └── zeekit.sh           # Bash script to run Zeek on mulitple files in a folder
│   ├── poison_cicids.py        # Script to run a poisoning attack on the CIC IDS 2018 dataset
│   ├── poison_ctu.py           # Script to run a poisoning attack on the CTU 13 dataset
│   ├── poison_ctu_ae.py        # Script to run a poisoning attack on the CTU 13 dataset with autoencoder features
│   └── poison_iscx.py          # Script to run a poisoning attack on the ISCX dataset
└── setup.py
```

## Use

The backdoor attacks can be run with the `scripts/poison_*.py` scripts.
For instance to perform a poisoning attack against a Gradient Boosting classifier on the CTU-13 Neris dataset, you can use the following:
```bash
python scripts/poison_ctu.py --model GradientBoosting --n_features 8 --seed 42 --fstrat entropy --subscenario 1;
```
This will run the attack considering the 8 most important features according to the Entropy feature selection strategy.

Examples of how to run the attacks can be found in the bash scripts in `scripts/bash`.


### Reproducing the results

To reproduce the results of the paper run the bash scripts listed below. 

To minimize the running time, for each figure, we suggest running the list of scripts in parallel (preferably on at least two machines).
Results of the attacks will be saved together with the poisoned data. Therefore, there is no need to run the same scripts more than once.

Figure 3:
- `scripts/bash/entropy_full.sh`
- `scripts/bash/gini_full.sh`
- `scripts/bash/shap_full.sh`
- `scripts/bash/random_full.sh`
- `notebooks/visualizations/ctu13_results.ipynb` (visualization)

Figure 4:
- `scripts/bash/entropy_gen.sh`
- `scripts/bash/entropy_red.sh`
- `notebooks/visualizations/ctu13_results.ipynb` (visualization)

Figure 5:
- `scripts/bash/entropy_gen.sh`
- `scripts/bash/shap_gen.sh`
- `notebooks/visualizations/poisoning_divergence_benign.ipynb` (visualization)

Figure 6:
- `scripts/bash/entropy_full_cicids.sh`
- `scripts/bash/shap_full_ccids.sh`
- `scripts/bash/entropy_full_iscx.sh`
- `scripts/bash/shap_full_iscx.sh`
- `notebooks/visualizations/other_results.ipynb` (visualization)

Table 4:
- `scripts/bash/entropy_ae_multi.sh`
- `notebooks/visualizations/ctu13_ae_results.ipynb` (visualization)