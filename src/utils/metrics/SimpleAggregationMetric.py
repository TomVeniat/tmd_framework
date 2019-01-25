import logging

from ignite.engine import Events
from ignite.metrics import Metric


class SimpleAggregationMetric(Metric):
    def __init__(self, update_fn, output_transform=lambda x: x):
        super(SimpleAggregationMetric, self).__init__(output_transform=output_transform)

        if not callable(update_fn):
            raise TypeError("Argument compute_fn should be callable")

        self.update_fn = update_fn
        self.value = None

    def reset(self):
        self.value = None

    def update(self, value):
        self.value = value if self.value is None else self.update_fn(self.value, value)

    def compute(self):
        return self.value

    def attach(self, engine, name):
        super(SimpleAggregationMetric, self).attach(engine, name)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name)
