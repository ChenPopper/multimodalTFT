import os

import torch
import torch.distributed as dist
import yaml
import numpy as np


def load_config(file):
    with open(file, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    return config


def dist_judge():
    if torch.distributed.is_initialized():
        # 分布式训练模式
        global_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else None

        if world_size > 1:
            # 多节点多卡训练
            print(
                f"Multi-node multi-GPU training. Global Rank: {global_rank}, World Size: {world_size}, Local Rank: {local_rank}")
        else:
            # 单机多卡训练
            print(
                f"Single-node multi-GPU training. Global Rank: {global_rank}, World Size: {world_size}, Local Rank: {local_rank}")
        return True
    else:
        # 非分布式训练
        print("Single-GPU training.")
        return False


def distributed():
    num_gpus = int(os.environ['WORLD_SIZE']) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    return distributed


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank():
    if 'LOCAL_RANK' not in os.environ:
        return get_rank()
    else:
        return int(os.environ['LOCAL_RANK'])


def synchronize():
    '''
    Helper function to synchronize among all processes when using distributed training
    '''
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def __mkdir__(file_name):
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except:
            pass


class FileUtils(object):
    def __init__(self):
        super().__init__()

    @staticmethod
    def makedir(dirs):
        if not os.path.exists(dirs):
            os.makedirs(dirs)

    @staticmethod
    def makefile(dirs, filename):
        f = open(os.path.join(dirs, filename), "a")
        f.close()

    def make_updir(self, file_name):
        dir_name = os.path.dirname(file_name)
        self.makedir(dir_name)


import time
import datetime


class TimeCounter:
    def __init__(self, start_epoch, num_epochs, epoch_iters):
        self.start_epoch = start_epoch
        self.num_epochs = num_epochs
        self.epoch_iters = epoch_iters
        self.start_time = None

    def reset(self):
        self.start_time = time.time()

    def step(self, epoch, batch):
        used = time.time() - self.start_time
        finished_batch_nums = (epoch - self.start_epoch) * self.epoch_iters + batch
        batch_time_cost = used / finished_batch_nums
        total = (self.num_epochs - self.start_epoch) * self.epoch_iters * batch_time_cost
        left = total - used
        return str(datetime.timedelta(seconds=left))


def folder_mkdir(file_path):
    if os.path.exists(file_path) is False and get_rank() == 0:
        try:
            os.makedirs(file_path)
        except:
            print(f'[LOCAL RANK {get_rank()}] {file_path} is  exist')


import shutil


def save_config(module_name, work_dir):
    '''
    Helper function to save the config setting
    '''
    log_path = os.path.join(work_dir, 'logs')
    folder_mkdir(work_dir)
    if get_rank() == 0:
        # datestr = datetime.datetime.now().strftime('%Y%m%d%H')
        datestr = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%Y%m%d%H")
        save_path = os.path.join(log_path, f"{datestr}_config.py")
        ori_config_path = module_name.replace('.', '/') + '.py'
        shutil.copyfile(ori_config_path, save_path)


def count_model_params(model):
    """Returns the total number of parameters of a PyTorch model

    Notes
    -----
    One complex number is counted as two parameters (we count real and imaginary parts)'
    """
    return sum(
        [p.numel() * 2 if p.is_complex() else p.numel() for p in model.parameters()]
    )


def mae_metric_4_ts(x, y, dim=()):
    abs_err = torch.abs(x - y)
    mae_err = torch.mean(abs_err, dim=dim)
    return mae_err


def update_dic_rec(dic, key, target_value):
    """to change the value to target_value of the given key
    recurrently
    example:
    dic = {
        'level_10': {
            'level_20': {'a': 2},
            'level_21': {'b': 3}
        },
        'level_11': {
            'level_20': {'a': 2},
            'level_21': {'c': 3}
        }
    }
    print(update_dic_rec(dic, 'a', 0))
    >> {'level_10': {'level_20': {'a': 0}, 'level_21': {'b': 3}}, 'level_11': {'level_20': {'a': 0}, 'level_21': {'c': 3}}}
    """
    if isinstance(dic, dict):
        for ele in dic.keys():
            if ele == key:
                dic[ele] = target_value
            else:
                update_dic_rec(dic[ele], key, target_value)
    return dic


def resolve_label_info(list_rec):
    """list_target : [tgt, pred, tgt_info]
    tgt_info: B, T, 1
    """
    import pandas as pd
    # timestamp = []
    # tc_name = []
    # gt_val = []
    # pred_val = []
    table = pd.DataFrame({})
    for tgt, pred, tgt_info in list_rec:
        data_dict = {'gt_value': tgt[0].cpu().numpy().squeeze(),
                     'pred_value': pred[0].cpu().numpy().squeeze(),
                     'lead_time': range(1, len(pred[0]) + 1),
                     'timestamp': [tgt_info[0][i][0][0] for i in range(len(tgt_info[0]))],
                     'name': [tgt_info[0][i][1][0].split('#&#')[0] for i in range(len(tgt_info[0]))]
                     # 'name': [tgt_info[0][i][1][0] for i in range(len(tgt_info[0]))]
                     }
        table = pd.concat([table, pd.DataFrame(data_dict)])
    return table.sort_values(['name', 'timestamp', 'lead_time'], ascending=True)


def init_weights(m, method='He'):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        if method == 'He':
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.uniform_(m.bias, a=0.001)
        elif method == 'Glorot':
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        else:
            pass


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def cross_entropy_loss(logits, labels, reduction='mean', label_smoothing=0.0):
    """
    logits: (B, n_class, n_channels)
    labels: (B, n_channels), dtype = torch.long
    """
    n_cls = logits.shape[1]
    label_one_hot = torch.nn.functional.one_hot(labels, n_cls).transpose(-2, -1)  # (B, n_class, n_channels)
    label_one_hot = label_one_hot.to(torch.float32)
    if label_smoothing > 0:
        label_one_hot *= 1 - label_smoothing
        label_one_hot += label_smoothing / n_cls

    cross_entropy = - torch.log(logits) * label_one_hot
    if reduction == 'mean':
        ce = torch.mean(torch.sum(cross_entropy))
    elif reduction == 'sum':
        ce = torch.sum(torch.sum(cross_entropy))
    return ce


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl
