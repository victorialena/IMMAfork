#!/bin/bash -ex

env=motion
model=imma
obs_frames=40
rollouts=9
randomseed=42
hidden_dim=256
edge_types=2

python main.py --env $env --model $model --randomseed $randomseed \
               --obs_frames $obs_frames --rollouts $rollouts --edge_types $edge_types \
               --num_epoch 200 --hidden_dim $hidden_dim --lr 1e-6 --plt
