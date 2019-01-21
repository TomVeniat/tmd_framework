import torch
from collections import namedtuple
import numpy as np

import torch.nn.functional as F

from ignite.engine import Events, Engine
from torch.distributions import Categorical

from src.experiments.ReinforceExperiment import RLExperiment

EPISODE_STARTED = Events.EPOCH_STARTED
EPISODE_COMPLETED = Events.EPOCH_COMPLETED

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class ActorCriticExperiment(RLExperiment):

    def select_action(self, model, observation):
        observation = torch.from_numpy(observation).float()
        probs, observation_value = model(observation)
        m = Categorical(probs)
        action = m.sample()
        model.saved_actions.append((m.log_prob(action), observation_value))
        return action.item()

    def finish_episode(self, model, optimizer, gamma, eps):
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