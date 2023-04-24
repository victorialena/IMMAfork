import math
import numpy as np
import os
import random
import sys
import torch


def prepare_dataset(args):
    if args.env == 'bball':
        if not args.long_term:
            all_data = np.load('./datasets/short_horizon_all_data.npy')
        else:
            all_data = np.load('./datasets/long_horizon_all_data.npy')
            all_data = all_data[:, ::2]

        x_min, x_max = all_data[:, :, :, 0].min(), all_data[:, :, :, 0].max()
        y_min, y_max = all_data[:, :, :, 1].min(), all_data[:, :, :, 1].max()
        scaling = [x_max, x_min, y_max, y_min]
        all_data[..., 0] = (all_data[..., 0] - x_min)/(x_max - x_min)
        all_data[..., 1] = (all_data[..., 1] - y_min)/(y_max - y_min)
        all_data[..., 5:, 3] = 1.
        all_data[..., -1, 2:] = 1.
        all_data = all_data[:args.dataset_size, ...]

        print('loaded all_data:', args.obs_frames)
        all_data, all_labels = all_data[:, :args.obs_frames, :, :], all_data[:, args.obs_frames:, :, :]
        all_data = torch.FloatTensor(all_data)
        all_labels = torch.FloatTensor(all_labels)
    else:
        assert False

    all_data = all_data[:, -args.obs_frames:, ...]
    
    print('data shape', all_data.shape)
    print('labels shape', all_labels.shape)

    dataset = torch.utils.data.TensorDataset(all_data, all_labels) # create your dataset
    train_size = int(len(dataset) * 0.8)
    val_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size - val_size
    torch.cuda.manual_seed_all(args.randomseed)
    torch.manual_seed(args.randomseed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Parameters
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 1,
              'pin_memory': False,
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
