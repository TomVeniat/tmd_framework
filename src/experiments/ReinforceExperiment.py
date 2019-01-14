import argparse
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

try:
    import gym
except ImportError:
    raise RuntimeError("Please install opengym: pip install gym")


from ignite.engine import Engine, Events



def select_action(model, observation):
    state = torch.from_numpy(observation).float().unsqueeze(0)
    probs = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.saved_log_probs.append(m.log_prob(action))
    return action.item()


EPISODE_STARTED = Events.EPOCH_STARTED
EPISODE_COMPLETED = Events.EPOCH_COMPLETED


class RLExperiment(object):

    def __init__(self, policy, gym_env, optim, device, gamma, nepisodes, render, log_interval):
        self.policy = policy
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
                print(self.policy.rewards)
                engine.terminate_epoch()
                # assert len(self.policy.rewards) - 1 == sum(self.policy.rewards)
                engine.state.last_episode_r = sum(self.policy.rewards)

        self.trainer = Engine(run_single_timestep)
        self.trainer._logger.setLevel(logging.WARNING)

        self.trainer.on(Events.STARTED)(self.initialize)
        self.trainer.on(EPISODE_STARTED)(self.reset_environment_state)
        self.trainer.on(EPISODE_COMPLETED)(self.update_model)
        self.trainer.on(EPISODE_COMPLETED)(self.log_episode)
        self.trainer.on(EPISODE_COMPLETED)(self.should_finish_training)

    def select_action(self, model, observation):
        observation = torch.from_numpy(observation).float()
        probs = model(observation)
        m = Categorical(probs)
        action = m.sample()
        model.saved_actions.append(m.log_prob(action))
        return action.item()

    def initialize(self, engine):
        engine.state.running_reward = 10

    def reset_environment_state(self, engine):
        engine.state.observation = self.gym_env.reset()

    def update_model(self, engine):
        r = engine.state.last_episode_r
        engine.state.running_reward = engine.state.running_reward * 0.99 + r * 0.01
        self.finish_episode(self.policy, self.optim, self.gamma, self.eps)

    def log_episode(self, engine):
        i_episode = engine.state.epoch
        if i_episode % self.log_interval == 0:
            print('Episode {}\tLast reward: {:>4.5}\tAverage reward: {:.2f}'.format(
                i_episode, engine.state.last_episode_r, engine.state.running_reward))

    def should_finish_training(self, engine):
        running_reward = engine.state.running_reward
        if running_reward > self.gym_env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, engine.state.last_episode_r))
            engine.should_terminate = True

    def run(self, _run):
        timesteps = list(range(10000))
        self.trainer.run(timesteps, max_epochs=self.nepisodes)

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
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        model.rewards = []
        model.saved_actions = []