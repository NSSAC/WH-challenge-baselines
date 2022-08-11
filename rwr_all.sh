#!/usr/bin/env bash

RWR_DIR=./sample_data/random_walks/
GRAPH=./sample_data/graph.pkl
N_JOBS=4

for (( PART=0; PART<$N_JOBS; PART++ ))
do
    mkdir -p $RWR_DIR/part_$PART 
    python ./rwr_all.py $GRAPH --pid_partition $PART --n_jobs $N_JOBS --output $RWR_DIR/part_$PART
done