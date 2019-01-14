import torch.nn as nn
import torch.nn.functional as F


class MnistClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(MnistClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class Policy(nn.Module):
    def __init__(self, obs_size, hidden_size, action_size, critic=False):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(obs_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_size)
        self.critic = critic
        if self.critic:
            self.value_head = nn.Linear(hidden_size, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        if self.critic:
            state_values = self.value_head(x)
            return F.softmax(action_scores, dim=-1), state_values

        return F.softmax(action_scores, dim=-1)
