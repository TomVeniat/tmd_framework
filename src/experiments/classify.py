import torch
import torch.optim as optim
import torch.nn.functional as F
from ignite._utils import convert_tensor as to
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage, Accuracy, Loss

from src.utils.metrics import Metrics
from .base import BaseExperiment

from tqdm import tqdm


def size(input_, dim=0):
    if isinstance(input_, torch.Tensor):
        return input_.shape[dim]
    elif isinstance(input_, collections.Mapping):
        return input_[next(iter(input_))].shape[dim]
    elif isinstance(input_, collections.Sequence):
        return input_[0].shape[dim]
    else:
        raise TypeError(("input must contain {}, dicts or lists; found {}"
                         .format(input_type, type(input_))))


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

        desc = "ITERATION - loss: {:.2f}"
        pbar = tqdm(
            initial=0, leave=False, total=len(self.train),
            desc=desc.format(0)
        )

        Accuracy(output_transform=lambda x: (x['y_pred'], x['y'])).attach(evaluator, 'acc')
        # RunningAverage(output_transform=lambda x: x['acc'])
        
        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            iter = (engine.state.iteration - 1) % len(self.train) + 1
            pbar.desc = desc.format(engine.state.output['loss'])
            pbar.update(1)
        
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(self.val)
            tqdm.write(
                "Validation Results - Epoch: {}  Avg accuracy: {:.2f}"
                .format(engine.state.epoch, evaluator.state.metrics['acc']))

            pbar.n = pbar.last_print_n = 0

        self.trainer = trainer
    

    def run(self, _run=None):
        self.trainer.run(self.train, max_epochs=self.nepochs)


# class MNISTExperiment(EpochExperiment):

#     def __init__(self, model, optim, train=[], val=[], test=[], **kwargs):
#         super(MNISTExperiment, self).__init__(**kwargs)
#         self.model = model
#         self.train = train
#         self.val = val
#         self.test = test
#         self.optim = optim

#     def init_metrics(self, *args, **kwargs):
#         m = Metrics(*args, **kwargs)
#         for name, _ in self.named_datasets():
#             m.Parent(name=name,
#                 children=(m.AvgMetric(name='loss'),
#                           m.AvgMetric(name='acc'))
#             )
#         m.Parent(name='state',
#             children=(m.AvgMetric(name='lr'),),
#         )
#         return m

#     def __call__(self, input, target, train=True, evaluate=True):
#         self.train_mode(train)
#         output = self.model(input)
#         loss = F.nll_loss(output, target)

#         if train:
#             self.optim.zero_grad()
#             loss.backward()
#             self.optim.step()

#         results = {}
#         if evaluate:
#             with torch.no_grad():
#                 self.train_mode(False)
#                 pred = output.max(1, keepdim=True)[1]
#                 correct = pred.eq(target.view_as(pred)).sum()
#                 results['acc'] = correct

#         return {
#             'loss': loss,
#             **results,
#         }