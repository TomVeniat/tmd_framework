from ignite.engine import Events
from ignite.metrics import Metric


class RunningAverage(Metric):
    def __init__(self, src=None, alpha=0.98):
        super(RunningAverage, self).__init__()

        if not isinstance(src, Metric):
            raise TypeError("Argument src should be a Metric or None")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("Argument alpha should be a float between 0.0 and 1.0")

        self.src = src
        self.alpha = alpha
        self._value = None

    def reset(self):
        self._value = None

    def update(self, output):
        # Implement abstract method
        pass

    def compute(self):
        if self._value is None:
            self._value = self._get_src_value()
        else:
            self._value = self._value * self.alpha + (1.0 - self.alpha) * self._get_src_value()
        return self._value

    def attach(self, engine, name):
        # Only init at the begin of the experiment
        engine.add_event_handler(Events.STARTED, self.started)

        # Update after each epoch/trajectory
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.completed, name)

    def _get_src_value(self):
        return self.src.compute()
