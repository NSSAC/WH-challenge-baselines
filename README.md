# Introduction

This repository contains example implementations for centralized models for the
Public Health use case for the US/UK PETS challenge. It contains code to generate
features from the synthetic data provided for competitors, as well as implementations
of example baselines. It additionally contains sample from the US network and
disease states to demonstrate data formatting expectations.

Note that the example data is just meant to demonstrate data formatting expectations
for these example implementations. Model performance for models trained just ont
this data will be quite poor. Replication of the performance metrics described
in the technical report requires running these example implementations on 
significantly larger portions of the data.

## Requirements 

The requirements for these implementations are listed in `requirements.txt`
Note that additional steps may be necessary if you wish to train the graph
neural network using GPUs.

# Sample Data

Examples of data files used in this implementation may be found in the 
`sample_data` directory. This contains a smaller contact file, `contacts.csv`,
representing a subset of the edges from the VA graph. This object is a csv 
containing a list of edges.

The other files in the `sample_data` directory are csv files `pop.csv`,
`training.csv` and `target.csv`. These files are subsets of the population, 
training and evaluation files containing only the persons present in the 
`graph.pkl` subgraph.

The directory `random_walks` contains example data produced by the `DeepInf/rwr_all.sh`
bash script. These are random walks on `graph.pkl` produced by `rwr_all.py`. 

The directories `sample_data/gnn` and `sample_data/logistic_regression`

## Logistic Regression

The logistic regression data generation, training, and evaluation can be executed
by going into the `logistic_regression` directory and executing the 
`./logistic_regression.sh` bash script. 

This generates the training and evaluation data for the logistic regression in
the `sample_data/logistic_regression/` directory, and performs the training and
evaluation of the data.

## GNN (DeepInf)

The GNN data generation can be done by going into the `DeepInf` directory, and
executing the `rwr_all.sh` bash script followed by the `make_data.sh` script. 
This generates the training and evaluation data for the Graph Neural Network 
based example implementation in `sample_data/gnn/train/`
and `sample_data/gnn/eval`

To train and evaluate the model, executing the `train_gnn.sh` bash script.
This creates checkpoint models in the `sample_data/gnn/models` and a csv with
predictions `sample_data/gnn/test_eval.csv`

The code for the graph neural network architecture is used from  

Jiezhong Qiu et al., “DeepInf: Social Influence Prediction with Deep Learning,” in Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2018, 2110–19, https://doi.org/10.1145/3219819.3220077.

# Parallelization of Data Preparation

To run these models on larger datasets, it may be desirable to parallelize certain
aspects of the data preparation. Much of the data preparation is embarrasingly
parallel. While the specifics of how this is parallelized depends on the platform,
we have written the data preparation scripts to suggest one possible approach.

The scripts `DeepInf/rwr_all.sh`, `DeepInf/make_data.sh', and 
`logistic_regression/logistic_regression.sh`, and all contain for-loops over 
`N_PART` iterations. The contents of these for-loops may be executed in parallel, 
and the number of jobs may be increased arbitrarily. 
