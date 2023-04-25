#!/bin/bash -ex

env=bball
model=imma
obs_frames=40
randomseed=42
hidden_dim=256
edge_types=5

python main.py --env $env --model $model --randomseed $randomseed \
               --obs_frames $obs_frames --rollouts 10 --edge_types $edge_types \
               --num_epoch 200 --hidden_dim $hidden_dim --lr 1e-6 --plt --dataset_size 100000

python main.py --env $env --model $model --randomseed $randomseed \
               --obs_frames $obs_frames --rollouts 20 --edge_types $edge_types \
               --num_epoch 200 --hidden_dim $hidden_dim --lr 1e-6 --plt --long_term --dataset_size 100000
