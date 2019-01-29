import json
import logging
import os
import re

import logger as metric_logger
import numpy as np
import torch
import visdom
from ignite.engine import Engine, Events
from supernets.networks.StochasticSuperNetwork import StochasticSuperNetwork
from torch.distributions import Categorical
import torch.nn.functional as F

from src.utils import external_resources as external, get_exp_name, VISDOM_CONF_PATH
from src.utils.metrics.MetaSlidingMetric import SlidingMetric
from src.utils.metrics.SimpleAggregationMetric import SimpleAggregationMetric

logger = logging.getLogger(__name__)


def compute_returns(rewards, gamma, next_value=0):
    returns = []
    last_r = next_value
    for r in rewards[::-1]:
        last_r = r + gamma * last_r
        returns.append(last_r)
    return torch.stack(returns[::-1])

def select_action(model, observation):
    action_logits, value = model(observation)

    m = Categorical(logits=action_logits)
    action = m.sample()
    model.actions_log_prob.append(m.log_prob(action))

    return action.item(), value

def sample_trajectories(model, env, obs, render):
    # obs = engine.state.observation

    rewards = []
    values = []

    done = False
    while not done:
        obs = torch.FloatTensor(obs).unsqueeze(0)
        action, value = select_action(model, obs)

        obs, reward, done, infos = env.step(action)

        if render:
            env.render()
        rewards.append(torch.tensor([reward]).float())
        values.append(value)

    return rewards, values


def optim_step(model, rewards, values, optimizer, gamma, normalize_reward, vf_loss_coef, eps):
    loss = 0

    rewards = compute_returns(rewards, gamma)

    if normalize_reward:
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    if model.critic:
        value_loss = F.smooth_l1_loss(torch.cat(values), rewards, reduction='none').mean()
        loss += value_loss * vf_loss_coef

        advantage = (rewards - torch.cat(values)).detach()
    else:
        advantage = rewards

    loss += -(torch.stack(model.actions_log_prob) * advantage).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return rewards.sum().item(), value_loss.item()



class PolicyGradientExperiment(object):

    def __init__(self, policy, gym_env, optim, device, gamma, normalize_reward, nepisodes, render, log_interval, vf_loss_coef, _run):
        self._run = _run
        self.exp_name = get_exp_name(_run.config, _run._id)

        use_visdom = os.path.isfile(VISDOM_CONF_PATH)
        if use_visdom:
            self.visdom_conf = external.load_conf(VISDOM_CONF_PATH)
            self.visdom_conf.update(env=self.exp_name)
        else:
            self.visdom_conf = None
        self._run.info['vis_url'] = "http://{server}:{port}/env/{env}".format(**self.visdom_conf)

        self.metric_logger = metric_logger.Experiment(self.exp_name, use_visdom=use_visdom, visdom_opts=self.visdom_conf,
                                                      time_indexing=False, xlabel='Episodes', log_git_hash=False)
        self.n_videos = 0

        self.loggers = {
            'rewards': ['last', 'sliding', 'running', 'target'],
            'values': ['value', 'return'],
            'lengths': ['lastl'],
            'loss': ['value_l']
        }
        for cat, labels in self.loggers.items():
            for label in labels:
                self.metric_logger.SimpleMetric(name=cat, tag=label)

        self.policy = policy
        # if isinstance(self.policy, StochasticSuperNetwork):
        #     self.policy.log_probas = []

        self.gym_env = gym_env
        self.optim = optim
        self.gamma = gamma
        self.normalize_reward = normalize_reward
        self.device = device
        self.nepisodes = nepisodes
        self.render = render
        self.eps = np.finfo(np.float32).eps.item()
        self.log_interval = log_interval
        self.vf_loss_coef = vf_loss_coef

        def run_optim_step(engine, iteration):

            rewards, values = sample_trajectories(self.policy, self.gym_env, engine.state.observation, self.render)

            returns, value_loss = optim_step(self.policy, rewards, values, self.optim, self.gamma, self.normalize_reward, self.vf_loss_coef, self.eps)

            return {'reward': sum(rewards),
                    'timestep': len(rewards),
                    'value_l': value_loss,
                    'value':sum(values).item(),
                    'return': returns}

        self.trainer = Engine(run_optim_step)
        # self.trainer._logger.setLevel(logging.WARNING)

        SlidingMetric(200, lambda x: np.mean(x), lambda x: x['reward'].item()).attach(self.trainer, 'sliding')

        SimpleAggregationMetric(lambda old, new: new, lambda x: x['timestep']).attach(self.trainer, 'lastl')
        SimpleAggregationMetric(lambda old, new: new, lambda x: x['reward'].item()).attach(self.trainer, 'last')
        SimpleAggregationMetric(lambda old, new: new, lambda x: x['value_l']).attach(self.trainer, 'value_l')
        SimpleAggregationMetric(lambda old, new: new, lambda x: x['value']).attach(self.trainer, 'value')
        SimpleAggregationMetric(lambda old, new: new, lambda x: x['return']).attach(self.trainer, 'return')

        def running_update(old, new, gamma=0.99):
            return old * gamma + (1.0 - gamma) * new
        SimpleAggregationMetric(running_update, lambda x: x['reward'].item()).attach(self.trainer, 'running')


        self.trainer.on(Events.STARTED)(self.initialize)

        #todo: move to update loop
        self.trainer.on(Events.ITERATION_STARTED)(self.reset_state)

        self.trainer.on(Events.ITERATION_COMPLETED)(self.log_hparams)

        self.trainer.on(Events.ITERATION_COMPLETED)(self.log_episode)
        self.trainer.on(Events.ITERATION_COMPLETED)(self.should_finish_training)

    def initialize(self, engine):
        engine.state.running_reward = None

    def reset_state(self, engine):
        engine.state.observation = self.gym_env.reset()
        self.policy.actions_log_prob = []
        if hasattr(self.policy, 'log_probas'):
            self.policy.log_probas = []

    def log_hparams(self, engine):
        engine.state.metrics['target'] = self.gym_env.spec.reward_threshold


    def update_model(self, engine):
        self.finish_episode(self.policy, self.optim, self.gamma, self.normalize_reward, self.eps)

    def log_episode(self, engine):
        i_episode = engine.state.iteration
        for metric, value in engine.state.metrics.items():
            self._run.log_scalar(metric, value, i_episode)

        for cat, labels in self.loggers.items():
            for label in labels:
                self.metric_logger.metrics[label][cat].update(engine.state.metrics[label])
        # self.metric_logger.Parent_Reward.update(**engine.state.metrics)
        self.metric_logger.log_with_tag(tag='*', reset=True)
        engine.state.running_reward = self.metric_logger.rewards_sliding
        engine.state.last_episode_r = self.metric_logger.rewards_last


        if i_episode % self.log_interval == 0:
            self._save_videos()
            logger.info('Episode {}\tLast reward: {:>4.5}\tAverage reward: {:.2f}'.format(
                i_episode, engine.state.last_episode_r, engine.state.running_reward))
            for met, met_val in engine.state.metrics.items():
                logger.info('\t\t{:8}:{:7.2f}'.format(met, met_val))
            logger.info("http://{server}:{port}/env/{env}".format(**self.visdom_conf))

    def should_finish_training(self, engine):
        running_reward = engine.state.running_reward
        if running_reward > self.gym_env.spec.reward_threshold:
            logger.info("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, engine.state.last_episode_r))
            engine.should_terminate = True
            # self.gym_env.close()

    def run(self):
        logger.info(self.policy)

        timesteps = list(range(self.nepisodes))
        self.trainer.run(timesteps, max_epochs=1)
        self.gym_env.unwrapped.close()
        return self.trainer.state.running_reward

    def _save_videos(self):
        if hasattr(self.gym_env, 'videos') and len(self.gym_env.videos) > self.n_videos:
            new_videos = self.gym_env.videos[self.n_videos:]
            for file_path, meta in new_videos:
                name = re.findall(".*\.video\..*\.(video[0-9]*.mp4)$", file_path)[0]
                with open(meta) as meta_file:
                    meta = json.load(meta_file)
                self._run.add_artifact(file_path, name, meta)
            self.n_videos += len(new_videos)
