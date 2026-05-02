import time
import cv2
import torch
from tqdm import tqdm
import pytorch_warmup as warmup
import numpy as np
import random
import os

from mfplnet.models.registry import build_net
from .registry import build_trainer, build_evaluator
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from mfplnet.datasets import build_dataloader
from mfplnet.utils.recorder import build_recorder
from mfplnet.utils.net_utils import save_model, load_network, resume_network
from mmcv.parallel import MMDataParallel
from thop import profile

import torch
import numpy as np
import random
from mmcv.parallel import MMDataParallel
from thop import profile

from mfplnet.models.utils.odconv import ODConv2d


class Runner(object):
    def __init__(self, cfg):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.best_metric = None
        self.cfg = cfg
        self.recorder = build_recorder(self.cfg)
        self.net = build_net(self.cfg)
        self.calculate_flops_and_params()
        self.net = MMDataParallel(self.net, device_ids=range(self.cfg.gpus)).cuda()

        assert next(self.net.parameters()).is_cuda, "Model parameters must be on GPU"

        self.recorder.logger.info('Network: \n' + str(self.net))
        self.resume()
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        self.metric = 0.
        self.val_loader = None
        self.test_loader = None

    def calculate_flops_and_params(self):
        resize_op = self.cfg.dataset.train.pipeline.transforms[0]
        assert resize_op['name'] == 'Resize', "First transform must be Resize"
        img_size = resize_op['parameters']['size']

        input_tensor = torch.randn(1, 3, img_size['height'], img_size['width']).cuda()

        self.net = self.net.cuda()
        self.net.eval()

        flops, params = profile(self.net, inputs=(input_tensor,))
        self.flops = flops / 1e9
        self.params = params / 1e6
        self.recorder.logger.info(f'FLOPs: {self.flops:.2f} GFLOPs')
        self.recorder.logger.info(f'Parameters: {self.params:.2f} M')

    def to_cuda(self, batch):
        for k in batch:
            if not isinstance(batch[k], torch.Tensor):
                continue
            batch[k] = batch[k].cuda()
        return batch

    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(self.net, self.cfg.load_from, finetune_from=self.cfg.finetune_from, logger=self.recorder.logger)

    def train_epoch(self, epoch, train_loader):
        self.net.train()
        start_time = time.time()
        max_iter = len(train_loader)
        self.recorder.logger.info(f'Total iterations: {self.cfg.total_iter}')
        self.recorder.logger.info(f'Actual iterations per epoch: {len(train_loader)}')
        for i, data in enumerate(train_loader):
            if self.recorder.step >= self.cfg.total_iter:
                break
            date_time = time.time() - start_time
            self.recorder.step += 1
            data = self.to_cuda(data)
            output = self.net(data)
            self.optimizer.zero_grad()
            loss = output['loss'].sum()
            loss.backward()
            self.optimizer.step()
            if not self.cfg.lr_update_by_epoch:
                self.scheduler.step()
            batch_time = time.time() - start_time
            self.recorder.update_loss_stats(output['loss_stats'])
            self.recorder.batch_time.update(batch_time)
            self.recorder.data_time.update(date_time)

            if i % self.cfg.log_interval == 0 or i == max_iter - 1:
                lr = self.optimizer.param_groups[0]['lr']
                self.recorder.lr = lr
                self.recorder.record('train')

        epoch_duration = time.time() - start_time
        fps = max_iter / epoch_duration
        self.recorder.logger.info(f'Epoch {epoch}: FPS: {fps:.2f}')


    def train(self):
        self.recorder.logger.info('Build train loader...')
        train_loader = build_dataloader(self.cfg.dataset.train, self.cfg, is_train=True)

        self.recorder.logger.info('Start training...')
        start_epoch = 0
        if self.cfg.resume_from:
            start_epoch = resume_network(self.cfg.resume_from, self.net, self.optimizer, self.scheduler, self.recorder)
        for epoch in range(start_epoch, self.cfg.epochs):
            self.recorder.epoch = epoch

            self.train_epoch(epoch, train_loader)
            if (epoch + 1) % self.cfg.save_ep == 0 or epoch == self.cfg.epochs - 1:
                self.save_ckpt()
            if (epoch + 1) % self.cfg.eval_ep == 0 or epoch == self.cfg.epochs - 1:
                self.validate()
            if self.recorder.step >= self.cfg.total_iter:
                break
            if self.cfg.lr_update_by_epoch:
                self.scheduler.step()

    def test(self):
        if not self.test_loader:
            self.test_loader = build_dataloader(self.cfg.dataset.test, self.cfg, is_train=False)
        self.net.eval()
        predictions = []
        start_time = time.time()
        for i, data in enumerate(tqdm(self.test_loader, desc=f'Testing')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                output = self.net.module.heads.get_lanes(output)
                predictions.extend(output)
            if self.cfg.view:
                self.test_loader.dataset.view(output, data['meta'])

        metric = self.test_loader.dataset.evaluate(predictions, self.cfg.work_dir)
        if metric is not None:
            self.recorder.logger.info('metric: ' + str(metric))

        test_duration = time.time() - start_time
        fps = len(self.test_loader) / test_duration
        self.recorder.logger.info(f'Test: FPS: {fps:.2f}, FLOPs: {self.flops:.2f} GFLOPs, Parameters: {self.params:.2f} M')


    def validate(self):
        if not self.val_loader:
            self.val_loader = build_dataloader(self.cfg.dataset.val, self.cfg, is_train=False)
        self.net.eval()
        predictions = []
        start_time = time.time()
        for i, data in enumerate(tqdm(self.val_loader, desc=f'Validate')):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                output = self.net.module.heads.get_lanes(output)
                predictions.extend(output)
            if self.cfg.view:
                self.val_loader.dataset.view(output, data['meta'])

        metric = self.val_loader.dataset.evaluate(predictions, self.cfg.work_dir)
        self.recorder.logger.info('metric: ' + str(metric))
        if self.best_metric is None:
            self.best_metric = metric
            self.save_ckpt(is_best=True)
        else:
            if (metric['Accuracy'] > self.best_metric['Accuracy']) or \
                    (metric['Accuracy'] == self.best_metric['Accuracy'] and
                     metric['Mean Iou'] > self.best_metric['Mean Iou']):
                self.best_metric = metric
                self.save_ckpt(is_best=True)

        val_duration = time.time() - start_time
        fps = len(self.val_loader) / val_duration
        self.recorder.logger.info(f'Validation: FPS: {fps:.2f}, FLOPs: {self.flops:.2f} GFLOPs, Parameters: {self.params:.2f} M')


    def save_ckpt(self, is_best=False):
        save_model(self.net, self.optimizer, self.scheduler, self.recorder, is_best)