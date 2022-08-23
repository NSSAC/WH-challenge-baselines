#!/usr/bin/env bash

mkdir -p ../sample_data/gnn/models
python train_gnn.py --tensorboard-log=gnn --model=gat --hidden-units=16,16 \
    --heads=8,8,1 --dim=64 --epochs=20 --lr=0.1 --dropout=0.2 \
    --file-dir=../sample_data/gnn/train/full --batch=200 --train-ratio=80 --valid-ratio=15 \
    --use-vertex-feature --instance-normalization --class-weight-balanced --shuffle \
    --model-save-file ../sample_data/gnn/models/gat 

python evaluate_model.py ../sample_data/gnn/models/gat_20.pt ../sample_data/gnn/eval/full ../sample_data/gnn/test_eval.csv --eval-labels ../sample_data/va_disease_outcome_target.csv

