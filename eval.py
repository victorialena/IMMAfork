import argparse
import numpy as np
import pandas as pd
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm
import pdb
from prepare_dataset import prepare_dataset
from losses import fde, kmin_ade, argsort_kade

from models.imma import IMMA



parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='springs5', choices=['bball', 'springs5', 'motion', 'ind'])
parser.add_argument('--baseline', type=str, default='imma')
parser.add_argument('--output_dir', type=str, default='logs')
parser.add_argument('--timesteps', type=int, default=50)
parser.add_argument('--obs_frames', type=int, default=40)
parser.add_argument('--input_size', type=int, default=4)
parser.add_argument('--randomseed', type=int, default=42)
parser.add_argument('--num_vars', type=int, default=5)
parser.add_argument('--prediction_steps', type=int, default=9, metavar='N', help='Num steps to predict before re-using teacher forcing.')


def unnormalize(data, data_max, data_min):
    return (data + 1) * (data_max - data_min) / 2. + data_min


def print_test_logs(suffix, mse_log, ade_log, fde_log, outfile=sys.stdout):
    print('mse_'+suffix+': {:.10f}'.format(np.mean(mse_log)),
          'ade_'+suffix+': {:.10f}'.format(np.mean(ade_log)),
          'fde_'+suffix+': {:.10f}'.format(np.mean(fde_log)),
          file=outfile)

def test_k(k=20, print_id=None, save_plot=False, save_csv=True):
    model.eval()
    
    mse_test = []
    ade_test = []
    fde_test = []

    for batch_idx, (batch_data, batch_label, *other_labels) in enumerate(test_loader):
        batch_data = batch_data.cuda()
        batch_label = batch_label.cuda()

        all_preds = []
        for _ in range(k):
            output = model.multistep_forward(batch_data, None, args.prediction_steps)
            output = torch.stack([p[-1] for p in output]).transpose(0,1)
            all_preds.append(output)

        all_preds = torch.stack(all_preds, dim=0)

        # undo preprocessing
        _output, _target = all_preds[..., :half_dims], batch_label[:, :args.prediction_steps, :, :half_dims]

        if args.env == 'nba':
            _output[..., 0] = unnormalize(_output[..., 0], loc_max, loc_min)
            _output[..., 1] = unnormalize(_output[..., 1], vel_max, vel_min)
            _target[..., 0] = unnormalize(_target[..., 0], loc_max, loc_min)
            _target[..., 1] = unnormalize(_target[..., 1], vel_max, vel_min)

        elif args.env.startswith('spring'):
            _output = unnormalize(_output, loc_max, loc_min)
            _target = unnormalize(_target, loc_max, loc_min)

        elif args.env.startswith('motion'):
            _output = unnormalize(_output, loc_max, loc_min)
            _target = unnormalize(_target, loc_max, loc_min)

        # pdb.set_trace()
        ade_values, ade_idx = kmin_ade(_output, _target, return_indices=True)
        ade_test.extend(ade_values)
        min_output = _output[ade_idx, range(_output.shape[1])] #args.batch_size)]
        mse_test.extend(F.mse_loss(min_output, _target, reduction='none').mean(-1).mean(-1))
        fde_test.extend(fde(min_output, _target))
        
        if (print_id is not None) and (print_id[0] == batch_idx):

            # if save_plot:
            #     if args.env == 'nba':
            #         from data.nba.visualize import plot_prediction
            #     elif args.env == 'springs5':
            #         from data.springs5.visualize import plot_prediction

            #     _, idx = kmin_ade(_output, _target, return_indices=True)
            #     sample_id = idx[print_id[1]]

            #     save_to = os.path.join(args.load_folder if run_only_test_phase else save_folder, 'figs')
            #     if not os.path.exists(save_to):
            #         os.makedirs(save_to)

            #     # file_name = args.baseline+'_'+args.env+'.png'
            #     file_name = args.baseline+'_'+args.env+'_'+str(print_id[0])+'_'+str(print_id[1])+'.png'

            #     plot_target = _target[print_id[1]].cpu().numpy()
            #     plot_output = _output[sample_id, print_id[1]].cpu().numpy()

            #     plot_output = np.concatenate((plot_target[:, -args.prediction_steps-1:-args.prediction_steps], plot_output), axis=1)
            #     plot_prediction(plot_output, plot_target, save_to, file_name=file_name)
                
            if save_csv:
                indices = argsort_kade(_output, _target)[:, print_id[1]]
                _output = torch.cat((batch_data[..., :half_dims].repeat(k, 1, 1, 1, 1), _output), dim=2)
                _target = torch.cat((batch_data[..., :half_dims], _target), dim=1)

                # T = args.prediction_steps
                plot_target = _target[print_id[1]]
                plot_target = plot_target.reshape(plot_target.size(0), -1).cpu().numpy()
                
                column_names = ['x', 'y'] if half_dims==2 else ['x', 'y', 'z']
                column_names = [_+str(i) for i in range(args.num_vars) for _ in column_names]
                df = pd.DataFrame(plot_target,
                                  columns = [_+'_gt' for _ in column_names])
                
                for i in indices:
                    traj = _output[i, print_id[1]]
                    traj = traj.reshape(traj.size(0), -1).cpu().numpy()
                    _df = pd.DataFrame(traj,
                                       columns = [_+'_'+str(i.item()) for _ in column_names])
                    df = df.join(_df)

                df.to_csv(args.output_dir+'/IMMA_'+args.env+'.csv')

            break

    mse_test = torch.stack(mse_test).cpu().numpy()
    ade_test = torch.stack(ade_test).cpu().numpy()
    fde_test = torch.stack(fde_test).cpu().numpy()

    idx = np.argmin(ade_test)
    batch_size = _target.shape[0]
    batch_id, sample_id = idx//batch_size, idx%batch_size
    
    print('<-----------------Testing----------------->')
    print('BEST SAMPLE: (batch)', batch_id, '(sample)', sample_id)
    print_test_logs('test', mse_test, ade_test, fde_test)



args = parser.parse_args()
half_dims = args.input_size//2

model_saved_name = '{}/best_model.pth'.format(args.output_dir)
model = torch.load(model_saved_name)

train_loader, valid_loader, test_loader, (loc_max, loc_min, vel_max, vel_min) = prepare_dataset(args)
with torch.no_grad():
    test_k(print_id=None) # nba (156, 2))# spring (35, 53))