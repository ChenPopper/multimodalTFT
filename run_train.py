import os
import sys
import torch
import torch.distributed as dist

from models import TropicalCyclone
from trainer import TCTrainer

from utils.common import distributed, get_local_rank
from dataset.utils import ZarrRead, load_json
from dataset.utils import times_select
from dataset import build_dataloader
from configs.data_config import (
    label_s_dict,
    date_setting,
    train_conf,
)


def load_data():
    data_file = 'data_file'
    mean_file = 'mean.nc'
    std_file = 'std.nc'


    plot_pictures = False
    data_vars = train_conf['src_data']['data_vars']

    label_file = 'label.json'
    label_dict = load_json(label_file)
    # print(label_dict)

    label_keys = ['key1', 'key2', 'key3', 'key4']

    label_train = label_dict
    label_train_dict, time_stamps = times_select(date_setting['begin'], date_setting['end'], label_train)

    train_data = ZarrRead(data_file, data_vars, time_stamps, mean_file, std_file)
    train_data.label = label_train_dict
    train_data.label_s_dict = label_s_dict
    out_vars = 'price'
    out_type = 'regression'

    train_ds = build_dataloader(
        cfg=train_conf,
        train_data=train_data, batch_size=train_conf['batch_size']
    )
    return train_ds


def main():
    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"

    train_loader = load_data()

    model = TropicalCyclone(train_conf)

    if distributed():
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[get_local_rank()],
                output_device=get_local_rank(),
                broadcast_buffers=False,
                find_unused_parameters=False
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_conf['lr'],
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=train_conf['train']['milestones'],
        gamma=train_conf['train']['scheduler_gamma']
    )
    trainer = TCTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=train_conf
    )

    trainer.train(
        train_loader,
        training_loss=torch.nn.SmoothL1Loss(),
    )


if __name__ == '__main__':

    if distributed():
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl',
                                world_size=world_size,
                                rank=rank,
                                init_method='env://')

    main()
