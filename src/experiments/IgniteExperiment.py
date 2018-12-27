import torch
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Loss, Accuracy


class IgniteExperiment(object):
    def __init__(self, model, trainset, testset, optim, device, nepochs):
        loss = torch.nn.NLLLoss()

        self.train_loader = trainset
        self.val_loader = testset
        self.nepochs = nepochs

        # print(model.device())
        model.to(device)

        self.trainer = create_supervised_trainer(model, optim, loss, device=device)
        self.evaluator = create_supervised_evaluator(model, device=device,
                                                     metrics={
                                                         'accuracy': Accuracy(),
                                                         'nll': Loss(loss)
                                                     })

        # self.trainer.on(Events.ITERATION_COMPLETED)(self.log_training_loss)
        self.trainer.on(Events.EPOCH_COMPLETED)(self.log_training_results)
        self.trainer.on(Events.EPOCH_COMPLETED)(self.log_validation_results)

    def log_training_loss(self, trainer):
        print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))

    def log_training_results(self, trainer):
        self.evaluator.run(self.train_loader)
        metrics = self.evaluator.state.metrics
        print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['accuracy'], metrics['nll']))

    def log_validation_results(self, trainer):
        self.evaluator.run(self.val_loader)
        metrics = self.evaluator.state.metrics
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['accuracy'], metrics['nll']))

    def run(self, _run):
        self.trainer.run(self.train_loader, max_epochs=self.nepochs)
