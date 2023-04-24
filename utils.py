import random
import os
import torch
import math

from tqdm import tqdm

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pdb
from datetime import date
import sys


def get_device(args):
    if args.gpu != -1:
        return torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def get_output_dir(args):
    save_folder = 'logs/'+args.env+'/'    
    base_str = date.today().strftime("%m-%d-%y")+'_exp'
    iter = max([int(file.replace('_', '.').split('.')[1][3:]) for file in os.listdir(save_folder) if base_str in file], default=0)
    return save_folder+base_str+str(iter+1)+'/'

    
def print_logs(epoch, suffix, nll_log, mse_log, ade_log, fde_log, outfile=sys.stdout):
    print('Epoch {}'.format(epoch+1),
          'nll_'+suffix+': {:.10f}'.format(np.mean(nll_log)),
          'mse_'+suffix+': {:.10f}'.format(np.mean(mse_log)),
          'ade_'+suffix+': {:.10f}'.format(np.mean(ade_log)),
          'fde_'+suffix+': {:.10f}'.format(np.mean(fde_log)),
          file=outfile)
    

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def set_random_seeds(seed):
    """
    Sets the random seeds for pytorch cpu and gpu
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None


def get_graph_accuracy(model, generator, args):
    if args.env == 'bball':
        return np.nan
    
    N = model.num_humans

    model.eval()
    acc = []
    loss_fn = lambda x, y : torch.mean(x==y).item() #torch.nn.BCELoss()

    with torch.no_grad():
        for iidx, (batch_data, _, gt_graphs) in tqdm(enumerate(generator)):
            batch_data = batch_data.to(args.device)
            gt_graphs = (gt_graphs.to(args.device) > 0.).to(float)

            batch_graph = None
            if args.gt:
                batch_graph = batch_data[:, -1, :, -N:]

            preds = model.multistep_forward(batch_data, batch_graph, 1)
            predicted_graphs = preds[0][0][0].to(float)
            mask = torch.ones(N, N) - torch.eye(N) == True
            pdb.set_trace()
            for g, gt in zip(predicted_graphs, gt_graphs):
                acc.append(loss_fn(torch.sigmoid(g.flatten()), gt[mask]).item())

    return np.mean(acc)

def get_mutual_info_score(model, generator, args):
    from sklearn.metrics import normalized_mutual_info_score
    num_humans = args.num_humans
    feat_dim = args.feat_dim
    device = args.device
    results = []
    for ii in range(args.edge_types):
        for jj in range(ii+1, args.edge_types):
            tmpa, tmpb = [], []
            for batch_data, batch_label in tqdm(generator):
                batch_graph = None
                batch_data = batch_data.to(device)
                batch_label = batch_label[:, :, :num_humans, :feat_dim].to(device)
                preds = model.multistep_forward(batch_data[:, -args.obs_frames:, ...],
                                                batch_graph, args.rollouts)

                for i in range(preds[0][0][0].shape[0]):
                    _, indices = torch.sort(preds[0][0][ii][i, 0, :], dim=-1)
                    indices = indices.detach().cpu().numpy().flatten()
                    new_indices = [0 for _ in range(num_humans-1)]
                    for j in range(num_humans-1):
                        new_indices[indices[j]] = j
                    tmpa.extend(new_indices)

                    _, indices = torch.sort(preds[0][0][jj][i, 0, :], dim=-1)
                    indices = indices.detach().cpu().numpy().flatten()
                    new_indices = [0 for _ in range(num_humans-1)]
                    for j in range(num_humans-1):
                        new_indices[indices[j]] = j
                    tmpb.extend(new_indices)
            results.append(normalized_mutual_info_score(tmpa, tmpb))

    return np.mean(results)

def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)