import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from src.utils.metrics import Metrics
from src.utils.torch import set_requires_grad
from src.experiments.base import BaseExperiment
from src.modules.losses import GANLoss


class UnpairedExperiment(BaseExperiment):

    def __init__(self, netG, netD,
                 train, test, netG_optim=None, netD_optim=None, lr_gen=None, lr_dis=None,
                 device='cuda', backloss_measurement=2., **kwargs):
        super(UnpairedExperiment, self).__init__(**kwargs)
        self.netG = netG
        self.netD = netD

        self.train = train
        self.test = test

        self.corruption = self.train.dataset.corruption

        self.netG_optim = netG_optim
        self.netD_optim = netD_optim
        self.backloss_measurement = backloss_measurement

        self.lr_gen = lr_gen
        self.lr_dis = lr_dis
        self.device = device

        self.likelihood_loss = nn.MSELoss()
        self.prior_loss = GANLoss().to(device)

        trainer = Engine(self.train_step)
        evaluate = Engine(self.eval_step)

        @trainer.on(Events.Epoch)
        def ev(engine):
            evaluator.run(val)
            print('ok')

        self.trainer = trainer
        self.evaluator = evaluator

        self.trainer.run(train)

    def train_step(self, batch):
        sample = batch['sample']
        target = batch['target']
        measured_sample = batch['measured_sample']
        mask = batch['mask']

        self.train_mode(train)
        fake_sample, measured_fake_sample, fake_sample_rec, measured_fake_sample_rec = \
            self.forward(target, measured_sample, mask)

        loss_G = self.backward_G(fake_sample, target, measured_fake_sample, measured_fake_sample_rec)
        loss_D = self.backward_D(sample, target, fake_sample)

        return {
            'loss_G': loss_G,
            'loss_D': loss_D,
        }

    @torch.no_grad()
    def eval_step(self, batch):
        sample = batch['sample']
        target = batch['target']
        measured_sample = batch['measured_sample']
        mask = batch['mask']
        output = self.forward(target, measured_sample, mask)

        return {
            'sample': output['sample'],
            'fake_sample': output['fake_sample'],
            'loss_G': loss_G,
            'loss_D': loss_D,
        }

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

    # def init_metrics(self, *args, **kwargs):
    #     m = Metrics(*args, **kwargs)
    #     for name, _ in self.named_datasets():
    #         m.Parent(name=name,
    #             children=(m.AvgMetric(name='loss_G'),m.AvgMetric(name='loss_D'),
    #                       m.AvgMetric(name='MSE'))
    #         )
    #     m.Parent(name='state',
    #         children=(m.AvgMetric(name='lr_gen'),m.AvgMetric(name='lr_dis')),
    #         )
    #     return m


