#!/usr/bin/env bash

DISEASE=../sample_data/training.csv
POP=../sample_data/pop.csv
N_JOBS=4

for (( PART=0; PART<N_JOBS; PART++ ))
do
    TRAIN_DIRNAME=../sample_data/gnn/train/data_$PART
    RWR_DIR=../sample_data/random_walks/part_$PART

    # Make training data

    mkdir -p $TRAIN_DIRNAME
    python ./data/extract_full.py $RWR_DIR $TRAIN_DIRNAME --sample_size 300000 --disease-file $DISEASE --pop-file $POP 
    python ./data/compose_edgelist.py $TRAIN_DIRNAME/adjacency_matrix.npy $TRAIN_DIRNAME/vertex_id.npy $TRAIN_DIRNAME/edges.elist
    deepwalk --format edgelist --input $TRAIN_DIRNAME/edges.elist --number-walks 40 --representation-size 64 --walk-length 10 --window-size 5 --workers 8 --output $TRAIN_DIRNAME/deepwalk.emb_64

    # Make evaluation data

    TEST_DIRNAME=../sample_data/gnn/eval/data_$PART
    mkdir -p $TEST_DIRNAME 

    python ./data/extract_full.py $RWR_DIR $TEST_DIRNAME --sample_size 300000 --test-set --disease-file $DISEASE --test-date-start 50 --pop-file $POP
    python ./data/compose_edgelist.py $TEST_DIRNAME/adjacency_matrix.npy $TEST_DIRNAME/vertex_id.npy $TEST_DIRNAME/edges.elist
    deepwalk --format edgelist --input $TEST_DIRNAME/edges.elist --number-walks 20 --representation-size 64 --walk-length 10 --window-size 5 --workers 8 --output $TEST_DIRNAME/deepwalk.emb_64
done

# Train and eval need access to the same embeddings and static node 
# features. This merges the data from each part directory into a single
# file and deduplicates the embeddings and static node features.

# If the scripts in the for-loop are "map" then these steps are "reduce"

FULL_TRAIN_DIR=../sample_data/gnn/train/full
FULL_EVAL_DIR=../sample_data/gnn/eval/full

mkdir -p $FULL_TRAIN_DIR
python ./data/collate.py ../sample_data/gnn/train $FULL_TRAIN_DIR

mkdir -p $FULL_EVAL_DIR
python ./data/collate.py ../sample_data/gnn/eval $FULL_EVAL_DIR

python ./data/merge_shared.py $FULL_TRAIN_DIR $FULL_EVAL_DIR $FULL_TRAIN_DIR $FULL_EVAL_DIR
