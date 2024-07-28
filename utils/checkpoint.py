import os
import torch
import torch.nn as nn
from collections import OrderedDict

from .common import get_rank


def __mkdir__(file_name):
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except:
            pass


def save_checkpoint(file_name, model, epoch=None, optimizer=None, scheduler=None):

    __mkdir__(file_name)
    save_dict = OrderedDict()
    if get_rank() == 0:
        if hasattr(model, 'module'):
            model = model.module
        save_dict['model'] = model.state_dict()
        if epoch is not None:
            save_dict['plan'] = dict(epoch=epoch)
        else:
            save_dict['plan'] = dict(save_stage='train_finished')
        if optimizer is not None:
            save_dict['optimizer'] = optimizer.state_dict()
        if scheduler is not None:
            save_dict['scheduler'] = scheduler.state_dict()
        torch.save(save_dict, file_name)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    # 加载checkpoint
    if hasattr(model, 'module'):
        model = model.module
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # 获取模型的参数
    checkpoint_model_keys = [ikey for ikey in checkpoint['model'].keys()]
    # check = checkpoint['model']
    model_dict = model.state_dict()
    check_ = OrderedDict()
    for num, (k, v) in enumerate(model_dict.items()):
        if k != checkpoint_model_keys[num]:
            print('stop')
        check_[k] = checkpoint['model'][checkpoint_model_keys[num]]
    model.load_state_dict(check_)
    # 获取优化器的参数
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    # 获取调度器的参数
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    if 'plan' in checkpoint:
        plan = checkpoint['plan']
    else:
        plan = None

    return model, optimizer, scheduler, plan

