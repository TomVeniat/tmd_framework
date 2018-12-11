import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from src.utils.metrics import Metrics
from .base import EpochExperiment
from ..modules.losses import GANLoss


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class UnpairedExperiment(EpochExperiment):

    def __init__(self, netG, netD,
                 trainset,  testset, netG_optim=None, netD_optim=None, lr_gen=None, lr_dis=None,
                 device='cuda', backloss_measurement=2., **kwargs):

        super(UnpairedExperiment, self).__init__(**kwargs)
        self.netG = netG
        self.netD = netD

        self.trainset = trainset
        self.testset = testset

        self.corruption = self.trainset.dataset.corruption

        self.netG_optim = netG_optim
        self.netD_optim = netD_optim
        self.backloss_measurement = backloss_measurement

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
                children=(m.AvgMetric(name='loss_G'),m.AvgMetric(name='loss_D'),
                          m.AvgMetric(name='MSE'))
            )
        m.Parent(name='state',
            children=(m.AvgMetric(name='lr_gen'),m.AvgMetric(name='lr_dis')),
            )
        return m

    def __call__(self, sample, target, measured_sample, mask, train=True, evaluate=True):
        self.train_mode(train)
        fake_sample, measured_fake_sample, fake_sample_rec, measured_fake_sample_rec = \
            self.forward(target, measured_sample, mask)

        if train:
            loss_G = self.backward_G(fake_sample, target, measured_fake_sample, measured_fake_sample_rec)
            loss_D = self.backward_D(sample, target, fake_sample)

        results = {}
        if evaluate:
            with torch.no_grad():
                self.train_mode(False)
                results['MSE'] = self.likelihood_loss(sample, fake_sample)
                results['loss_G'] = loss_G
                results['loss_D'] = loss_D

        return results

    def forward(self, target, measured_sample, mask):
        fake_sample = self.netG(measured_sample, target=target, mask=mask)
        measurement = self.corruption.measure(measured_sample)
        measured_fake_sample = measurement["measured_sample"]
        theta = measurement['theta']

        fake_sample_rec = self.netG(measured_fake_sample, target=target,
                                             mask=theta)

        measurement_rec = self.corruption.measure(fake_sample_rec, theta=theta)
        measured_fake_sample_rec = measurement_rec["measured_sample"]

        return fake_sample, measured_fake_sample, fake_sample_rec, measured_fake_sample_rec

    def backward_G(self, fake_sample, target, measured_fake_sample, measured_fake_sample_rec):
        set_requires_grad(self.netD, False)
        self.netG_optim.zero_grad()

        pred_fake = self.netD(fake_sample, target=target)
        loss_G = self.prior_loss(pred_fake, True)

        loss_back = self.likelihood_loss(measured_fake_sample, measured_fake_sample_rec.detach())
        loss_G = loss_G + loss_back * self.backloss_measurement

        loss_G.backward()
        self.netG_optim.step()
        return loss_G

    def backward_D(self, sample, target, fake_sample):
        set_requires_grad(self.netD, True)
        self.netD_optim.zero_grad()

        pred_real = self.netD(sample, target=target)
        pred_fake = self.netD(fake_sample.detach(), target=target)

        # Real
        loss_D_real = self.prior_loss(pred_real, True)
        # Fake
        loss_D_fake = self.prior_loss(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        self.netD_optim.step()

        return loss_D


