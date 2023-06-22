import math
import numpy as np
import os
import random
import sys
import torch
import pdb

def split_fn(dataset, random):
    N = len(dataset)
    train_size, val_size = int(0.8*N), int(0.1*N)
    test_size = N - train_size - val_size
    if random:
        return torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    return (torch.utils.data.Subset(dataset, range(train_size)),
            torch.utils.data.Subset(dataset, range(train_size, train_size+val_size)),
            torch.utils.data.Subset(dataset, range(train_size+val_size, train_size+val_size+test_size)))

dataset_size=100000

def prepare_dataset(args):
    if args.env == 'bball':
        all_data = np.load('./datasets/long_horizon_all_data.npy')
        all_data = all_data[:, ::2]

        all_data[..., 5:, 3] = 1.
        all_data[..., -1, 2:] = 1.
        all_data = all_data[:dataset_size, ...]
        gt_edges = np.ones((dataset_size, 11, 11))
        gt_edges[:, np.arange(11), np.arange(11)] = 0
        gt_edges = torch.FloatTensor(gt_edges)
        idx = 2
    
    elif args.env == 'springs5':
        all_data = np.moveaxis(np.load('./datasets/springs5_all_data.npy'), 1, 2)
        gt_edges = torch.FloatTensor(np.load('./datasets/springs5_edges.npy'))
        idx = 2

    elif args.env == 'motion':
        all_data = np.load('./datasets/motion_features.npy')
        src, dst = np.load('./datasets/motion_edges.npy').T

        batch_sz, T, NUM_JOINTS, d = all_data.shape
        gt_edges = torch.zeros((NUM_JOINTS, NUM_JOINTS))
        gt_edges[src, dst] = 1
        gt_edges = gt_edges.repeat(batch_sz, 1, 1)
        idx = 3
    else:
        assert False

    if args.normalize:
        x_min, x_max = all_data[:, :, :, :idx].min(), all_data[:, :, :, :idx].max()
        v_min, v_max = all_data[:, :, :, idx:].min(), all_data[:, :, :, idx:].max()
        scaling = [x_max, x_min, v_max, v_min]
        all_data[..., :idx] = (all_data[..., :idx] - x_min)/(x_max - x_min)
        all_data[..., idx:] = (all_data[..., idx:] - v_min)/(v_max - v_min)
    else:
        scaling = None

    all_data, all_labels = all_data[:, :args.obs_frames, :, :], all_data[:, args.obs_frames:args.obs_frames+args.rollouts, :, :]
    all_data = torch.FloatTensor(all_data)
    all_labels = torch.FloatTensor(all_labels)
    
    # print('loaded all_data:', all_data.size(0))
    # print('data shape', all_data.shape)
    # print('labels shape', all_labels.shape)
    dataset = torch.utils.data.TensorDataset(all_data, all_labels, gt_edges) # create your dataset

    torch.cuda.manual_seed_all(args.randomseed)
    torch.manual_seed(args.randomseed)
    train_dataset, val_dataset, test_dataset = split_fn(dataset, False) # args.env == 'bball')

    # Parameters
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 1,
              'pin_memory': False,
              'drop_last': False,
              }
    train_generator = torch.utils.data.DataLoader(train_dataset, **params)
    val_generator = torch.utils.data.DataLoader(val_dataset, **params)
    test_generator = torch.utils.data.DataLoader(test_dataset, **params)
    return train_generator, val_generator, test_generator, scaling


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser('')
#     parser.add_argument('--env', type=str, default='bball', choices=['socialnav', 'phase', 'bball'])
#     parser.add_argument('--config', type=str, default='configs/default.py')
#     parser.add_argument('--dataset_size', type=int, default=300000)
#     parser.add_argument('--randomseed', type=int, default=42)
#     parser.add_argument('--obs_frames', type=int, default=40)
#     parser.add_argument('--rollouts', type=int, default=10)
#     args = parser.parse_args()
#     prepare_dataset(args)
