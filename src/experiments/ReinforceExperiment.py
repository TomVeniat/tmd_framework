import argparse
import logging
import os

import networkx as nx
import numpy as np
import logger as metric_logger


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from supernets.networks.StochasticSuperNetwork import StochasticSuperNetwork

import matplotlib.pyplot as plt
import gym


from ignite.engine import Engine, Events

from src.utils import external_resources as external, get_exp_name, VISDOM_CONF_PATH

# from src.utils.metrics.RunningAverageMetric import RunningAverage
from src.utils.metrics.MetaSlidingMetric import MetaSlidingMetric
from src.utils.metrics.RunningAverageMetric import RunningAverage
from src.utils.metrics.SimpleAggregationMetric import SimpleAggregationMetric

logger = logging.getLogger(__name__)

EPISODE_STARTED = Events.EPOCH_STARTED
EPISODE_COMPLETED = Events.EPOCH_COMPLETED


class RLExperiment(object):

    def __init__(self, policy, gym_env, optim, device, gamma, nepisodes, render, log_interval, _run):
        self._run = _run
        self.exp_name = get_exp_name(_run.config, _run._id)

        use_visdom = os.path.isfile(VISDOM_CONF_PATH)
        if use_visdom:
            visdom_conf = external.load_conf(VISDOM_CONF_PATH)
            visdom_conf.update(env=self.exp_name)
        else:
            visdom_conf = None

        self.metric_logger = metric_logger.Experiment(self.exp_name, use_visdom=use_visdom, visdom_opts=visdom_conf,
                                                      time_indexing=False, xlabel='Episodes', log_git_hash=False)

        metrics = ['slidingreward', 'runningaverage', 'reward', 'length']
        self.metric_logger.ParentWrapper(tag='Reward', name='Parent',
                                children=[self.metric_logger.SimpleMetric(name=metric) for metric in metrics])

        self.policy = policy
        if isinstance(self.policy, StochasticSuperNetwork):
            self.policy.set_probas(torch.ones(1, self.policy.n_stoch_nodes))
            self.policy.log_probas = []


        self.gym_env = gym_env
        self.optim = optim
        self.gamma = gamma
        self.device = device
        self.nepisodes = nepisodes
        self.render = render
        self.eps = np.finfo(np.float32).eps.item()
        self.log_interval = log_interval

        def run_single_timestep(engine, timestep):
            observation = engine.state.observation
            action = self.select_action(self.policy, observation)
            engine.state.observation, reward, done, _ = self.gym_env.step(action)
            if self.render:
                self.gym_env.render()
            self.policy.rewards.append(reward)

            if done:
                engine.terminate_epoch()
                engine.state.last_episode_r = sum(self.policy.rewards)
                engine.state.last_episode_l = timestep
            return {'reward': torch.tensor([reward]).float(), 'timestep': torch.tensor([timestep]), 'done': done}

        self.trainer = Engine(run_single_timestep)
        self.trainer._logger.setLevel(logging.WARNING)

        # avg_output = RunningAverage(output_transform=lambda x: x[0])

        r_metric = SimpleAggregationMetric(update_fn=lambda old, new: old + new, output_transform=lambda x: (x['reward']))
        r_metric.attach(self.trainer, 'reward')
        s_metric = MetaSlidingMetric(win_size=100, update_fn=lambda x: np.mean(x), src_metric=r_metric)
        s_metric.attach(self.trainer, 'slidingreward')
        RunningAverage(r_metric, 0.99,).attach(self.trainer, 'runningaverage')

        SimpleAggregationMetric(lambda old, new: new, lambda x: x['timestep']).attach(self.trainer, 'length')
        # RunningAverage(None, 0.99, lambda x: x['reward']).attach(self.trainer, 'running_averageDirect')

        self.trainer.on(Events.STARTED)(self.initialize)
        self.trainer.on(EPISODE_STARTED)(self.reset_environment_state)
        self.trainer.on(EPISODE_COMPLETED)(self.update_model)
        self.trainer.on(EPISODE_COMPLETED)(self.should_finish_training)

        def metrics_printer(engine):
            logger.info(engine.state.metrics)

        # self.trainer.on(EPISODE_COMPLETED)(metrics_printer)
        self.trainer.on(EPISODE_COMPLETED)(self.log_episode)



    def select_action(self, model, observation):
        observation = torch.from_numpy(observation).float().unsqueeze(0)

        if isinstance(self.policy, StochasticSuperNetwork):
            x = [observation, torch.zeros(1, self.policy.action_size)]
            probs = model(x)[0]
            m = Categorical(logits=probs)
        else:
            x = observation
            probs = model(x)
            m = Categorical(probs)
        action = m.sample()
        model.saved_actions.append(m.log_prob(action))
        return action.item()

    def initialize(self, engine):
        engine.state.running_reward = None

    def reset_environment_state(self, engine):
        engine.state.observation = self.gym_env.reset()

    def update_model(self, engine):
        r = engine.state.last_episode_r
        old_r = engine.state.running_reward
        engine.state.running_reward = r if old_r is None else old_r * 0.99 + r * 0.01
        self.finish_episode(self.policy, self.optim, self.gamma, self.eps)

    def log_episode(self, engine):
        i_episode = engine.state.epoch
        for metric, value in engine.state.metrics.items():
            self._run.log_scalar(metric, value, i_episode)
        self.metric_logger.Parent_Reward.update(**engine.state.metrics)
        self.metric_logger.log_with_tag(tag='*', reset=True)

        if i_episode % self.log_interval == 0:
            logger.info('Episode {}\tLast reward: {:>4.5}\tAverage reward: {:.2f}'.format(
                i_episode, engine.state.last_episode_r, engine.state.running_reward))
            logger.info('\tmetrics:{}'.format(engine.state.metrics))

    def should_finish_training(self, engine):
        running_reward = engine.state.running_reward
        if running_reward > self.gym_env.spec.reward_threshold:
            logger.info("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, engine.state.last_episode_r))
            engine.should_terminate = True
            # self.gym_env.close()

    def run(self):
        logger.info(self.policy)

        timesteps = list(range(10000))
        self.trainer.run(timesteps, max_epochs=self.nepisodes)
        self.gym_env.unwrapped.close()
        return self.trainer.state.running_reward

    def finish_episode(self, model, optimizer, gamma, eps):
        R = 0
        policy_loss = []
        rewards = []
        for r in model.rewards[::-1]:
            R = r + gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        for log_prob, reward in zip(model.saved_actions, rewards):
            policy_loss.append(-log_prob * reward)
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        model.rewards = []
        model.saved_actions = []
