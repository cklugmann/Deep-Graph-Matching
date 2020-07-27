import os
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils import data

from datasets.keypoints_dataset import Dataset
from model.modules.voting_layer import distance_loss
from model.model import save_weights

from utils.visualization_utils import show_images_and_keypoints
from utils.metrics_utils import pck_metric


class Trainer:
    def __init__(self, model, train_conf, data_conf):
        self.data_conf = data_conf
        self.train_conf = train_conf

        self.trainloader = data.DataLoader(
            Dataset(data_conf, mode='train'),
            batch_size=train_conf.batch_size_train, shuffle=True
        )
        self.testloader = data.DataLoader(
            Dataset(data_conf, mode='val'),
            batch_size=train_conf.batch_size_val, shuffle=False
        )
        self.model = model
        opt_settings = self.train_conf.optimizer
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=opt_settings['learning_rate'],
                                    betas=(opt_settings['beta_a'],
                                           opt_settings['beta_b']))

        self.subdir = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        if not os.path.exists(self.get_exp_dir()):
            os.makedirs(self.get_exp_dir())
            os.makedirs(os.path.join(self.get_exp_dir(), 'imgs'))
        if not os.path.exists(self.train_conf.save_weights_path):
            os.makedirs(self.train_conf.save_weights_path)

    def transform_batch(self, batch):
        if self.is_cuda():
            return tuple([x.cuda() for x in batch])
        return batch

    def train_step(self, batch, step_id=0):
        self.model.train()
        batch = self.transform_batch(batch)
        self.optimizer.zero_grad()
        dist_pred, _ = self.model(batch)
        loss = distance_loss(dist_pred, *batch[-3:])
        loss.backward()
        self.optimizer.step()
        return loss.item(), step_id+1

    def train_epoch(self, step_id=0, benchmark_pck=0.):
        for n_it, batch in enumerate(self.trainloader):
            train_loss, step_id = self.train_step(batch, step_id=step_id)
            if step_id % 10 == 0:
                print('Train {} {}'.format(step_id, train_loss))
                # print(torch.sum(self.model.affinity.lambda1.grad[:4, :4]))
                self.write_to_log(step_id, train_loss, mode='train')
            if step_id % 20 == 0:
                val_loss, pck = self.eval_model(step_id=step_id, write_image=(step_id % 200 == 0))
                print('Val {} {} {}'.format(step_id, val_loss, pck))
                self.write_to_log(step_id, val_loss, pck=pck, mode='val')
                if pck > benchmark_pck:
                    benchmark_pck = pck
                    save_weights(self.model, save_path=self.train_conf.save_weights_path)

            if n_it >= self.train_conf.iters_per_epoch - 1:
                break

        return step_id, benchmark_pck

    def train_epochs(self, number_epochs=1):
        step_id = 0
        benchmark_pck = 0.
        for _ in range(number_epochs):
            step_id, benchmark_pck = self.train_epoch(step_id=step_id, benchmark_pck=benchmark_pck)

    def eval_model(self, step_id=0, write_image=False):
        self.model.eval()
        batch = next(iter(self.testloader))
        batch = self.transform_batch(batch)
        with torch.no_grad():
            dist_pred, pos_pred = self.model(batch)
            loss = distance_loss(dist_pred, *batch[-3:])
            pck = pck_metric(batch[-2], pos_pred, batch[-1])
        if write_image:
            num_images = min(self.train_conf.batch_size_val,
                             self.train_conf.num_images_to_print)
            tmp = [b[:num_images] for b in batch[-3:]]
            show_images_and_keypoints(batch[0][:num_images],
                                      batch[1][:num_images],
                                      *tmp,
                                      pos_pred[:num_images],
                                      file=os.path.join(self.get_exp_dir(), 'imgs', 'out{}.pdf'.format(step_id)))
        return loss.item(), pck

    def write_to_log(self, step_id, loss, pck=None, mode='train'):
        mode = mode if mode in ['train', 'val'] else 'train'
        items_to_be_written = [str(step_id), str(loss)]
        if mode == 'val' and pck is not None:
            items_to_be_written.append(str(pck))
        with open(os.path.join(self.get_exp_dir(), '{}_loss.txt'.format(mode)), 'a+') as file:
            file.write('\t'.join(items_to_be_written))
            file.write('\n')

    def is_cuda(self):
        return next(self.model.parameters()).is_cuda

    def get_exp_dir(self):
        return os.path.join(self.train_conf.experiments_base_path, self.subdir)
