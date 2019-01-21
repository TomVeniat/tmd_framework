from ignite.engine import Events
from ignite.metrics import Metric


class MetaMetric(Metric):
    def __init__(self, src_metric=None, output_transform=None):
        super(MetaMetric, self).__init__(output_transform=output_transform)
        if not (isinstance(src_metric, Metric) or src_metric is None):
            raise TypeError("Argument src_metric should be a Metric or None")

        self.src_metric = src_metric

    def update(self, output):
        # Implement abstract method
        pass

    def attach(self, engine, name):
        if self.src_metric is None:
            # Behave as a normal metric
            super(MetaMetric, self).attach(engine, name)
        else:
            # Bind this metric as a Meta metric

            # Only init at the begin of the experiment
            engine.add_event_handler(Events.STARTED, self.started)

            # Update after each epoch/trajectory
            engine.add_event_handler(Events.EPOCH_COMPLETED, self.completed, name)

    def _get_src_value(self):
        return self.src_metric.compute()
