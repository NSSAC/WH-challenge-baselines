#!/usr/bin/env bash

N_JOBS=4
DISEASE=../sample_data/training.csv
POP=../sample_data/pop.csv
RWR_DIR=../sample_data/random_walks/
N_JOBS=4

mkdir -p ../sample_data/logistic_regression/train/
mkdir -p ../sample_data/logistic_regression/eval/

for (( PART=0; PART<$N_JOBS; PART++ ))
do
    python make_logistic_data.py $RWR_DIR/part_$PART $DISEASE $POP ../sample_data/logistic_regression/train/train_$PART.csv
    python make_logistic_data.py $RWR_DIR/part_$PART $DISEASE $POP ../sample_data/logistic_regression/eval/eval_$PART.csv --min-date 45 --is-eval
done

python logistic_regression.py --training_dir ../sample_data/logistic_regression/train/ --eval_dir ../sample_data/logistic_regression/eval/ \
    --pop $POP --eval-labels ../sample_data/target.csv
