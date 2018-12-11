import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from src.utils.metrics import Metrics
from .base import EpochExperiment
from ..modules.losses import GANLoss

class UnpairedExperiment(EpochExperiment):

    def __init__(self, netG, netD, netG_optim, netD_optim, lr_gen=None, lr_dis=None,
                 trainset=[],  testset=[], device='cuda', backloss_measurement=2., **kwargs):
        super(UnpairedExperiment, self).__init__(**kwargs)
        self.netG = netG
        self.netD = netD
        self.netG_optim = netG_optim
        self.netD_optim = netD_optim
        self.backloss_measurement = backloss_measurement

        self.train = trainset
        self.testset = testset

        self.likelihood_loss = nn.MSELoss()
        self.prior_loss = GANLoss().to(device)


        self.lr_gen = lr_gen
        self.lr_dis = lr_dis
        self.device = device

    def update_state(self, epoch):
        if self.lr_scheduler==None:
            lr = self.optim.param_groups[-1]['lr']
        else:
            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_lr()
            assert(len(lr) == 1)
            lr = lr[0]
        return {'lr': lr}

    def init_metrics(self, *args, **kwargs):
        m = Metrics(*args, **kwargs)
        for name, _ in self.named_datasets():
            m.Parent(name=name,
                children=(m.AvgMetric(name='loss'),
                          m.AvgMetric(name='acc'))
            )
        m.Parent(name='state',
            children=(m.AvgMetric(name='lr'),),
        )
        return m

    def __call__(self, sample, target, measured_sample, train=True, evaluate=True):
        self.train_mode(train)


        output = self.model(input)
        loss = F.nll_loss(output, target)

        if train:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        results = {}
        if evaluate:
            with torch.no_grad():
                self.train_mode(False)
                pred = output.max(1, keepdim=True)[1]
                correct = pred.eq(target.view_as(pred)).sum()
                results['acc'] = correct

        return {
            'loss': loss,
            **results,
        }
