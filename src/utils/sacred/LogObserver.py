from __future__ import division, print_function, unicode_literals

import logging
import pprint
from os import system

import visdom
from dateutil import tz
from sacred.observers.base import RunObserver

from src.utils import get_exp_name

logger = logging.getLogger(__name__)


def utc_to_local(datetime):
    utc = datetime.replace(tzinfo=tz.tzutc())
    return utc.astimezone(tz.tzlocal())


class LogObserver(RunObserver):

    @staticmethod
    def create(visdom_opts, *args, **kwargs):
        return LogObserver(visdom_opts, *args, **kwargs)

    def __init__(self, visdom_opts, *args, **kwargs):
        super(LogObserver, self).__init__(*args, **kwargs)
        self.visdom_opts = visdom_opts
        self.viz = None
        self.config = None
        self.run_id = None
        self.exp_name = None

    def started_event(self, ex_info, command, host_info, start_time, config, meta_info, _id):
        self.config = config
        self.run_id = _id
        self.exp_name = get_exp_name(config, _id)

        self.viz = visdom.Visdom(**self.visdom_opts, env=self.exp_name)

        # Ugly trick to set the pane name if using GNU Screen
        system("echo '\ek{}\e\\'".format(_id))

        logger.info(pprint.pformat(self.config))
        local_time = utc_to_local(start_time)
        self.viz.text('Started at {}'.format(local_time))

        logger.info("http://{server}:{port}/env/{env}".format(**self.visdom_opts, env=self.exp_name))

    def heartbeat_event(self, info, captured_out, beat_time, result):
        pass

    # def log_metrics(self, metrics_by_name, info):
    #     logger.info('')
    #     logger.info('')
    #     logger.info(f'Metric "{metrics_by_name}": {info}')
    #     logger.info('')
    #     logger.info('')


    def completed_event(self, stop_time, result):
        local_time = utc_to_local(stop_time)
        logger.info('completed_event')

        self.viz.text('Completed at {}'.format(local_time))

    def interrupted_event(self, interrupt_time, status):
        local_time = utc_to_local(interrupt_time)
        logger.info('interrupted_event')

        self.viz.text('Interruped at {}'.format(local_time))

    def failed_event(self, fail_time, fail_trace):
        local_time = utc_to_local(fail_time)
        logger.info('failed_event')

        self.viz.text('Failed at {}\n{}'.format(local_time, fail_trace))
