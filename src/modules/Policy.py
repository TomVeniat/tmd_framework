import torch

import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, obs_size, hidden_size, action_size, critic=False):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(obs_size, hidden_size)
        self.affine2 = nn.Linear(hidden_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_size)
        self.critic = critic
        if self.critic:
            self.value_head = nn.Linear(hidden_size, 1)

        self.saved_actions = []
        self.rewards = []
        # self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.affine1.weight)
        for p in self.parameters():
            if p.dim() >= 2:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.xavier_uniform_(p.view(1, -1))

    def forward(self, x):
        z1 = F.relu(self.affine1(x))
        z2 = F.relu(self.affine2(z1))
        action_logits = self.action_head(z2)
        value = self.value_head(z2) if self.critic else None
        return action_logits, value
