from ignite.engine import Events
from ignite.metrics import Metric

from src.utils.metrics.MetaMetric import MetaMetric


class MetaSlidingMetric(MetaMetric):
    def __init__(self, win_size, update_fn, src_metric=None, output_transform=None):
        super(MetaSlidingMetric, self).__init__(src_metric=src_metric, output_transform=output_transform)
        if not callable(update_fn):
            raise TypeError("Argument compute_fn should be callable")

        self.update_fn = update_fn
        self.win_size = win_size
        self.value = None

    def reset(self):
        self.value = []

    def compute(self):
        self.value.append(self._get_src_value())
        if len(self.value) > self.win_size:
            self.value.pop(0)

        return self.update_fn(self.value)


class SlidingMetric(Metric):
    def __init__(self, win_size, update_fn, output_transform=None):
        super(SlidingMetric, self).__init__(output_transform=output_transform)
        if not callable(update_fn):
            raise TypeError("Argument compute_fn should be callable")

        self.update_fn = update_fn
        self.win_size = win_size
        self.value = None

    def reset(self):
        self.value = []

    def update(self, output):
        self.value.append(output)
        if len(self.value) > self.win_size:
            self.value.pop(0)

    def compute(self):
        return self.update_fn(self.value)

    def attach(self, engine, name):
        super(SlidingMetric, self).attach(engine, name)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name)