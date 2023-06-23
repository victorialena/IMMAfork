#!/bin/bash -ex

model=imma
obs_frames=40
rollouts=9
randomseed=42
hidden_dim=256
edge_types=2 #5
num_epoch=100

# env=bball
# python main.py --env $env --model $model --randomseed $randomseed \
#                --obs_frames $obs_frames --rollouts $rollouts --edge_types $edge_types \
#                --num_epoch $num_epoch --hidden_dim $hidden_dim --lr 1e-6 --plt --long_term --dataset_size 100000


env=springs5
python main.py --env $env --model $model --randomseed $randomseed \
               --obs_frames $obs_frames --rollouts $rollouts --edge_types $edge_types \
               --num_epoch $num_epoch --hidden_dim $hidden_dim --lr 1e-6 --plt


env=motion
python main.py --env $env --model $model --randomseed $randomseed \
               --obs_frames $obs_frames --rollouts $rollouts --edge_types $edge_types \
               --num_epoch $num_epoch --hidden_dim $hidden_dim --lr 1e-6 --plt
