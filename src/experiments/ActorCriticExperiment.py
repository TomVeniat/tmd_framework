from collections import namedtuple
import numpy as np

import torch
import torch.nn.functional as F

from ignite.engine import Events, Engine
from torch.distributions import Categorical

EPISODE_STARTED = Events.EPOCH_STARTED
EPISODE_COMPLETED = Events.EPOCH_COMPLETED

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


def select_action(model, observation):
    observation = torch.from_numpy(observation).float()
    probs, observation_value = model(observation)
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append((m.log_prob(action), observation_value))
    return action.item()


def finish_episode(model, optimizer, gamma, eps):
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
        # print()
        # print('reward: {}'.format(reward))
        # print('L1: {}'.format(F.smooth_l1_loss(value, torch.tensor([r]))))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


class ActorCriticExperiment(object):

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
            action = select_action(self.policy, observation)
            engine.state.observation, reward, done, _ = self.gym_env.step(action)
            if self.render:
                self.gym_env.render()
            self.policy.rewards.append(reward)

            if done:
                engine.terminate_epoch()
                engine.state.timestep = timestep

        self.trainer = Engine(run_single_timestep)

        self.trainer.on(Events.STARTED)(self.initialize)
        self.trainer.on(EPISODE_STARTED)(self.reset_environment_state)
        self.trainer.on(EPISODE_COMPLETED)(self.update_model)
        self.trainer.on(EPISODE_COMPLETED)(self.log_episode)
        self.trainer.on(EPISODE_COMPLETED)(self.should_finish_training)

    def initialize(self, engine):
        engine.state.running_reward = 10

    def reset_environment_state(self, engine):
        engine.state.observation = self.gym_env.reset()

    def update_model(self, engine):
        t = engine.state.timestep
        engine.state.running_reward = engine.state.running_reward * 0.99 + t * 0.01
        finish_episode(self.policy, self.optim, self.gamma, self.eps)

    def log_episode(self, engine):
        i_episode = engine.state.epoch
        if i_episode % self.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, engine.state.timestep, engine.state.running_reward))

    def should_finish_training(self, engine):
        running_reward = engine.state.running_reward
        if running_reward > self.gym_env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, engine.state.timestep))
            engine.should_terminate = True

    def run(self, _run):
        timesteps = list(range(10000))
        self.trainer.run(timesteps, max_epochs=self.nepisodes)