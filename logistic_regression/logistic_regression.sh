#!/usr/bin/env bash

DISEASE=../sample_data/va_disease_outcome_training.csv
POP=../sample_data/va_person.csv
GRAPH=../sample_data/va_population_network.csv

N_JOBS=4

mkdir -p ../sample_data/logistic_regression/train/
mkdir -p ../sample_data/logistic_regression/eval/

for (( PART=0; PART < N_JOBS; PART++ ))
do
    python make_logistic_data.py $GRAPH $DISEASE $POP ../sample_data/logistic_regression/train/train_$PART.csv --pid_partition $PART --n_jobs $N_JOBS
    python make_logistic_data.py $GRAPH $DISEASE $POP ../sample_data/logistic_regression/eval/eval_$PART.csv --min-date 50 --is-eval --pid_partition $PART --n_jobs $N_JOBS
done
python logistic_regression.py --training_dir ../sample_data/logistic_regression/train/ --eval_dir ../sample_data/logistic_regression/eval/ \
    --pop $POP --eval-labels ../sample_data/va_disease_outcome_target.csv
