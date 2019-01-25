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

from src.utils import external_resources as external, get_exp_name, VISDOM_CONF_PATH
from src.utils.metrics.MetaSlidingMetric import SlidingMetric
from src.utils.metrics.SimpleAggregationMetric import SimpleAggregationMetric

logger = logging.getLogger(__name__)


class PolicyGradientExperiment(object):

    def __init__(self, policy, gym_env, optim, device, gamma, normalize_reward, nepisodes, render, log_interval, _run):
        self._run = _run
        self.exp_name = get_exp_name(_run.config, _run._id)

        use_visdom = os.path.isfile(VISDOM_CONF_PATH)
        if use_visdom:
            self.visdom_conf = external.load_conf(VISDOM_CONF_PATH)
            self.visdom_conf.update(env=self.exp_name)
        else:
            self.visdom_conf = None

        self.metric_logger = metric_logger.Experiment(self.exp_name, use_visdom=use_visdom, visdom_opts=self.visdom_conf,
                                                      time_indexing=False, xlabel='Episodes', log_git_hash=False)
        self.n_videos = 0

        self.loggers = {
            'rewards': ['last', 'sliding', 'running', 'target'],
            'lengths': ['lastl']
        }
        for cat, labels in self.loggers.items():
            for label in labels:
                self.metric_logger.SimpleMetric(name=cat, tag=label )


        self.policy = policy
        if isinstance(self.policy, StochasticSuperNetwork):
            self.policy.log_probas = []


        self.gym_env = gym_env
        self.optim = optim
        self.gamma = gamma
        self.normalize_reward = normalize_reward
        self.device = device
        self.nepisodes = nepisodes
        self.render = render
        self.eps = np.finfo(np.float32).eps.item()
        self.log_interval = log_interval

        def run_optim_step(engine, iteration):
            obs = engine.state.observation
            done = False

            while not done:
                obs = torch.FloatTensor(obs).unsqueeze(0)
                action, value = self.select_action(self.policy, obs)

                obs, reward, done, infos = self.gym_env.step(action)


                if self.render:
                    self.gym_env.render()
                self.policy.rewards.append(reward)

            return {'reward': sum(self.policy.rewards), 'timestep': len(self.policy.rewards)}

        self.trainer = Engine(run_optim_step)
        # self.trainer._logger.setLevel(logging.WARNING)

        SlidingMetric(200, lambda x: np.mean(x), lambda x: x['reward']).attach(self.trainer, 'sliding')

        SimpleAggregationMetric(lambda old, new: new, lambda x: x['timestep']).attach(self.trainer, 'lastl')
        SimpleAggregationMetric(lambda old, new: new, lambda x: x['reward']).attach(self.trainer, 'last')

        def running_update(old, new, gamma=0.99):
            return old * gamma + (1.0 - gamma) * new
        SimpleAggregationMetric(running_update, lambda x: x['reward']).attach(self.trainer, 'running')


        self.trainer.on(Events.STARTED)(self.initialize)

        #todo: move to update loop
        self.trainer.on(Events.ITERATION_STARTED)(self.reset_environment_state)

        self.trainer.on(Events.ITERATION_COMPLETED)(self.update_model)
        self.trainer.on(Events.ITERATION_COMPLETED)(self.log_hparams)

        self.trainer.on(Events.ITERATION_COMPLETED)(self.log_episode)
        self.trainer.on(Events.ITERATION_COMPLETED)(self.should_finish_training)

    def select_action(self, model, observation):
        action_logits, value = model(observation)

        m = Categorical(logits=action_logits)
        action = m.sample()
        model.saved_actions.append(m.log_prob(action))
        return action.item(), value

    def initialize(self, engine):
        engine.state.running_reward = None

    def reset_environment_state(self, engine):
        engine.state.observation = self.gym_env.reset()

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

    def compute_returns(self, rewards, gamma, next_value=0):
        returns = []
        last_r = next_value
        for r in rewards[::-1]:
            last_r = r + gamma * last_r
            returns.append(last_r)
        return returns[::-1]

    def finish_episode(self, model, optimizer, gamma, normalize_reward, eps):
        rewards = self.compute_returns(model.rewards, gamma)
        rewards = torch.tensor(rewards)
        if normalize_reward:
            rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

        policy_loss = [(-log_prob * reward) for log_prob, reward in zip(model.saved_actions, rewards)]
        # for log_prob, reward in zip(model.saved_actions, rewards):
        #     policy_loss.append(-log_prob * reward)
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        model.rewards = []
        model.saved_actions = []

    def _save_videos(self):
        if hasattr(self.gym_env, 'videos') and len(self.gym_env.videos) > self.n_videos:
            new_videos = self.gym_env.videos[self.n_videos:]
            for file_path, meta in new_videos:
                name = re.findall(".*\.video\..*\.(video[0-9]*.mp4)$", file_path)[0]
                with open(meta) as meta_file:
                    meta = json.load(meta_file)
                self._run.add_artifact(file_path, name, meta)
            self.n_videos += len(new_videos)
