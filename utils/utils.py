import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from typing import Tuple


def get_mask_per_node(num_nodes, num_vars, output_dim, input_dim, device, parallel=True, var=0):
    if parallel:
        mask_all = []
        neg_mask_all = []
        for var in range(num_vars):
            mask = torch.ones(num_nodes, dtype=torch.long, device=device)
            mask = torch.reshape(mask, (-1, num_vars))
            mask[:, var] = 0
            mask = torch.reshape(mask, (-1,))

            neg_mask = torch.lt(mask, 1)

            mask = torch.gt(mask, 0)
            mask = torch.unsqueeze(mask, 1)
            mask = mask.expand(-1, input_dim)

            mask_all.append(mask)
            neg_mask_all.append(neg_mask)

        mask_all = torch.cat(mask_all)
        neg_mask_all = torch.cat(neg_mask_all)

        return mask_all, neg_mask_all
    else:
        mask = torch.ones(num_nodes, dtype=torch.long, device=device)
        mask = torch.reshape(mask, (-1, num_vars))
        mask[:, var] = 0
        mask = torch.reshape(mask, (-1,))

        neg_mask = torch.lt(mask, 1)
        neg_mask = torch.unsqueeze(neg_mask,1)
        neg_mask = neg_mask.expand(-1,output_dim)

        mask = torch.gt(mask, 0)
        mask = torch.unsqueeze(mask, 1)
        mask = mask.expand(-1, input_dim)

        return mask, neg_mask

def get_global_labels(sensor_labels,num_vars,collapsed=False):
    '''
    Converts sensor-level anomaly labels into global-level anomaly labels of the same shape
    :param sensor_labels: tensor of sensor level labels
    :param num_vars: number of variables in the multi-variate system
    :param collapsed: if collapsed, the non-expanded labels are returned
    :return: tensor of global level labels expanded to each sensor
    '''
    target_glob = torch.reshape(sensor_labels, [-1, num_vars])
    target_glob = torch.any(target_glob, dim=1).long()
    if collapsed:
        return target_glob
    else:
        target_glob = torch.unsqueeze(target_glob, -1)
        target_glob = target_glob.expand(-1, num_vars)
        target_glob = torch.reshape(target_glob, (-1,))
        return target_glob


def getnormals(data,target,num_vars,node_labels=False):
    # Get indices of normal rows from target
    target = torch.round(target)
    if node_labels:
        target_n = get_global_labels(target,num_vars)
        target_n = torch.abs(target_n - 1)
        target_n = torch.gt(target_n, 0)
    else:
        target_n = torch.abs(target - 1)
        target_n = torch.unsqueeze(target_n, 1)
        target_n = target_n.expand(-1, num_vars)
        target_n = target_n.reshape(torch.numel(target_n))
        target_n = torch.gt(target_n, 0)
    # Keep only normal samples
    data_n = data[target_n, :]
    return data_n


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


'''
Utitily functions from https://github.com/wagner-d/TimeSeAD/tree/master/timesead
'''
def coe_batch(x: torch.Tensor, y: torch.Tensor, coe_rate: float, suspect_window_length: int) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Contextual Outlier Exposure.

    Args:
        x : Tensor of shape (batch, time, D)
        y : Tensor of shape (batch, )
        coe_rate : Number of generated anomalies as proportion of the batch size.
    """

    if coe_rate == 0:
        raise ValueError(f"coe_rate must be > 0.")
    batch_size, window_size, ts_channels = x.shape
    oe_size = int(batch_size * coe_rate)
    device = x.device

    # Select indices
    idx_1 = torch.randint(low=0, high=batch_size, size=(oe_size,), device=device)
    # sample and apply nonzero offset to avoid fixed points
    idx_2 = torch.randint(low=1, high=batch_size, size=(oe_size,), device=device)
    idx_2 += idx_1
    torch.remainder(idx_2, batch_size, out=idx_2)

    if ts_channels > 3:
        numb_dim_to_swap = torch.randint(low=3, high=ts_channels, size=(oe_size,), device=device)
    else:
        numb_dim_to_swap = torch.ones(oe_size, dtype=torch.long, device=device) * ts_channels

    x_oe = x[idx_1].clone()  # .detach()
    oe_time_start_end = torch.randint(low=x.shape[1] - suspect_window_length, high=x.shape[1] + 1, size=(oe_size, 2),
                                      device=device)
    oe_time_start_end.sort(dim=1)
    # for start, end in oe_time_start_end:
    for i in range(oe_size):
        # obtain the dimensions to swap
        numb_dim_to_swap_here = numb_dim_to_swap[i].item()
        dims_to_swap_here = torch.from_numpy(np.random.choice(ts_channels, size=numb_dim_to_swap_here, replace=False))\
            .to(torch.long).to(device)

        # obtain start and end of swap
        start, end = oe_time_start_end[i]
        start, end = start.item(), end.item()

        # swap
        x_oe[i, start:end, dims_to_swap_here] = x[idx_2[i], start:end, dims_to_swap_here]

    # Label as positive anomalies
    y_oe = torch.ones(oe_size, dtype=y.dtype, device=device)

    return x_oe, y_oe


def mixup_batch(x: torch.Tensor, y: torch.Tensor, mixup_rate: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x : Tensor of shape (batch, time, D)
        y : Tensor of shape (batch, )
        mixup_rate : Number of generated anomalies as proportion of the batch size.
    """

    if mixup_rate == 0:
        raise ValueError(f"mixup_rate must be > 0.")
    batch_size = x.shape[0]
    mixup_size = int(batch_size * mixup_rate)  #
    device = x.device

    # Select indices
    idx_1 = torch.randint(low=0, high=batch_size, size=(mixup_size,), device=device)
    # sample and apply nonzero offset to avoid fixed points
    idx_2 = torch.randint(low=1, high=batch_size, size=(mixup_size,), device=device)
    idx_2 += idx_1   
    torch.remainder(idx_2, batch_size, out=idx_2)

    # sample mixing weights:
    beta_param = 0.05
    weights = torch.from_numpy(np.random.beta(beta_param, beta_param, (mixup_size,))).type_as(x).to(device)
    oppose_weights = 1.0 - weights

    # Create contamination
    x_mix_1 = x[idx_1]
    x_mix_2 = x[idx_2]  # Pretty sure that the index here should be idx_2 instead of idx_1
    x_mixup = x_mix_1 * weights[:, None, None]
    x_mixup.addcmul_(x_mix_2, oppose_weights[:, None, None])

    # Label as positive anomalies
    y_mixup = y[idx_1] * weights
    y_mixup.addcmul_(y[idx_2], oppose_weights)

    return x_mixup, y_mixup

def calc_causal_same_pad(kernel_size: int, stride: int = 1, in_shape: int = 1, dilation: int = 1) -> int:
    return in_shape * (stride - 1) - stride + dilation * (kernel_size - 1) + 1


def calc_same_pad(kernel_size: int, stride: int = 1, in_shape: int = 1, dilation: int = 1) -> Tuple[int, int]:
    total_pad = calc_causal_same_pad(kernel_size, stride, in_shape, dilation)
    pad_start = total_pad // 2
    pad_end = total_pad - pad_start

    return pad_start, pad_end

activations = {
    'relu': torch.nn.ReLU(),
    'sigmoid': torch.nn.Sigmoid(),
    'tanh': torch.nn.Tanh(),
    'linear': torch.nn.Identity()
}