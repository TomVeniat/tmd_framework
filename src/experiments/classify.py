import torch
import torch.nn.functional as F
from ignite._utils import convert_tensor as to
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage, Accuracy, Loss

from src.utils.tensor import size
from .base import BaseExperiment


class MNISTExperiment(BaseExperiment):

    def train_step(self, engine, batch):
        self.training()
        input, target = to(batch, self.device)
        self.optimizer.zero_grad()
        output = self.model(input)
        loss = F.nll_loss(output, target)
        loss.backward()
        self.optimizer.step()
        return {
            'loss': loss.item(),
            'y_pred': output,
            'y': target
        }

    @torch.no_grad()
    def eval_step(self, engine, batch):
        input, target = to(batch, self.device)
        output = self.model(input)
        loss = F.nll_loss(output, target)
        return {
            'loss': loss.item(),
            'y_pred': output,
            'y': target
        }

    def __init__(self, model, optimizer, train=[], val=[], nepochs=10, use_tqdm=False, **kwargs):
        super(MNISTExperiment, self).__init__(**kwargs)
        self.model = model
        self.train = train
        self.val = val
        self.optimizer = optimizer
        self.nepochs = nepochs
        trainer = Engine(self.train_step)
        evaluator = Engine(self.eval_step)
        Accuracy(output_transform=lambda x: (x['y_pred'], x['y'])).attach(evaluator, 'acc')
        
        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            iter = (engine.state.iteration - 1) % len(self.train) + 1
            print("ITERATION - loss: {:.2f}".format(engine.state.output['loss']))
        
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(self.val)
            print(
                "Validation Results - Epoch: {}  Avg accuracy: {:.2f}"
                .format(engine.state.epoch, evaluator.state.metrics['acc']))

        self.trainer = trainer
    

    def run(self, _run=None):
        self.trainer.run(self.train, max_epochs=self.nepochs)