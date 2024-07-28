import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from timeit import default_timer
from utils.checkpoint import save_checkpoint
from utils.common import (
    __mkdir__,
    distributed,
    get_local_rank,
    TimeCounter,
)


# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s')
# logger = logging.getLogger(__name__)

use_gpu = torch.cuda.is_available()
device = "cuda" if use_gpu else "cpu"


# def draw_loss(loss_all, epoch, cfg):
#     plt.figure()
#     plt.plot(range(len(loss_all)), loss_all)
#     plt.xlabel('Batches')
#     plt.ylabel('Loss')
#     plt.title('Training Loss')
#     plt.title(f'Training Loss Epoch {epoch}')
#     s_name = os.path.join(cfg.work_dir, 'logs', 'imgs', 'loss', 'loss_img_epoch{}.png'.format(epoch))
#     __mkdir__(s_name)
#     plt.savefig(s_name)
#     plt.close()
#     cfg.logger.info(f'[ Save Loss imgs Successed Epoch {epoch} ] ||| {s_name} ')


class TCTrainer:
    """showing the log while training

    """
    def __init__(
            self,
            model,
            optimizer,
            scheduler,
            config,
            device=torch.device('cpu'),
            save_ckp=False  # to save model at checkpoint or not, default False
    ):

        self.callbacks = None
        self.save_dir = config['work_dir']
        self.save_name = os.path.join(self.save_dir, config['model_save_name'])
        self.save_ckp = save_ckp
        self.device = device
        self.model = model.to(device)
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = config['logger']

    def train(
            self,
            train_loader,
            # test_loader,
            training_loss=None,
            eval_losses=None,
    ):

        if training_loss is None:
            training_loss = torch.nn.MSELoss()

        if eval_losses is None:  # By default just evaluate on the training loss
            eval_losses = dict(l2=training_loss)

        errors = None
        epoch_batch_num = len(train_loader)
        time_counter = TimeCounter(0, self.config['n_epochs'], epoch_batch_num)
        time_counter.reset()
        self.logger.info('{:#^75}'.format(' Data Information '))
        self.logger.info(
            f"[DATA INFO] ||| [All Batch Num] {len(train_loader.batch_sampler)} "
            f"||| [Batch Size] {self.config['batch_size']} "
        )
        self.logger.info('{:#^75}'.format(' Data Information '))

        pbar = tqdm(range(self.config['n_epochs']), dynamic_ncols=True, smoothing=0.1)
        for epoch in pbar:

            self.model.train()

            loss_all, avg_loss = self.train_one_epoch(train_loader, epoch, time_counter, training_loss, eval_losses)

            if epoch % self.config['log_epoch_interval'] == 0 or epoch == self.config['n_epochs']-1:
                self.logger.info(
                    "In epoch {epoch}, the average loss = {avg_loss:.6f} ||| "
                    "MaxMemory = {max_mem:.1f}.".format(
                        epoch=epoch,
                        avg_loss=avg_loss,
                        max_mem=torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
                    )
                )

            if self.save_ckp:
                if epoch % self.config['ckpt_interval'] == 0:
                    save_file = os.path.join(
                        self.config['work_dir'], 'model', 'model_epoch_{}.pth'.format(epoch)
                    )
                    save_checkpoint(save_file, self.model, epoch)
                    self.logger.info(f"Checkpoint saved for epoch = {epoch}!")

        __mkdir__(self.save_name)
        save_checkpoint(self.save_name, self.model)  # save model at the end of training
        self.logger.info(f"Trained model is save in {self.save_name}")

        # return errors

    def train_one_epoch(
            self,
            train_loader,
            # test_loader,
            epoch,
            time_counter,
            training_loss=None,
            eval_losses=None,
    ):
        self.model.train()
        loss_all = []
        avg_loss = 0

        # track number of training examples in batch
        n_samples = 0
        for batch_num, (in_data, label, _) in enumerate(tqdm(train_loader)):
            n_samples += label.shape[0]
            in_data = in_data[0].cuda() if use_gpu else in_data.cpu()
            # in_data = in_data[0]
            label = label.cuda() if use_gpu else label.cpu()
            out = self.model(in_data, [in_data])
            loss = training_loss(out, label)
            del out

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_num % self.config['display_interval'] == 0:
                self.logger.info(
                    '    [LOCAL_RANK %d Epoch %d Batch %d ] ||| [lr: %.4f] [Loss: %.4f] ||| LeftTime %s MaxMemory %dMB' %
                    (get_local_rank(),
                     epoch,
                     batch_num,
                     self.optimizer.param_groups[0]['lr'],
                     loss.item(),
                     time_counter.step(epoch + 1, batch_num),
                     torch.cuda.max_memory_allocated(self.device) / 1024 ** 2))

            if batch_num % self.config['save_interval'] == 0 and batch_num > 0:
                save_file = os.path.join(
                    self.config['work_dir'], 'model', 'model_epoch{}_batch{}.pth'.format(epoch, batch_num)
                )
                save_checkpoint(save_file, self.model, epoch)
                self.logger.info('    [Save Model] model_epoch{}_batch{}.pth'.format(epoch, batch_num) + ' Saved')
            loss_all.append(loss.item())
            with torch.no_grad():
                avg_loss += loss.item()

        avg_loss = avg_loss / n_samples

        return loss_all, avg_loss

    def evaluate(self, loss_dict, data_loader, log_prefix=""):
        """Evaluates the model on a dictionary of losses

        Parameters
        ----------
        loss_dict : dict of functions
          each function takes as input a tuple (prediction, ground_truth)
          and returns the corresponding loss
        data_loader : data_loader to evaluate on
        log_prefix : str, default is ''
            if not '', used as prefix in output dictionary

        Returns
        -------
        errors : dict
            dict[f'{log_prefix}_{loss_name}] = loss for loss in loss_dict
        """

        self.model.eval()

        errors = {f"{log_prefix}_{loss_name}": 0 for loss_name in loss_dict.keys()}

        n_samples = 0
        with torch.no_grad():
            for idx, sample in enumerate(data_loader):
                sample = {
                    k: v.to(self.device)
                    for k, v in sample.items()
                    if torch.is_tensor(v)
                }
                n_samples += sample["y"].size(0)

                out = self.model(**sample)

                for loss_name, loss in loss_dict.items():
                    val_loss = loss(out, **sample)
                    if val_loss.shape == ():
                        val_loss = val_loss.item()

                    errors[f"{log_prefix}_{loss_name}"] += val_loss

        for key in errors.keys():
            errors[key] /= n_samples

        del out

        return errors
