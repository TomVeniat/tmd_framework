import torch

import torch.nn.functional as F
from supernets.interface.NetworkBlock import NetworkBlock, DummyBlock
from supernets.networks.StochasticSuperNetwork import StochasticSuperNetwork
from torch import nn


class Output(nn.Module):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, hidden_size, out_size, bias):
        super(Output, self).__init__()
        self.lin = nn.Linear(hidden_size, out_size, bias)

    def forward(self, inputs):
        assert isinstance(inputs, list)
        return self.lin(sum(inputs))


class LinearBlock(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, n_in, n_out, bias, relu):
        super(LinearBlock, self).__init__()
        self.lin = nn.Linear(n_in, n_out, bias=bias)
        self.relu = relu

    def forward(self, x):
        y = self.lin(x)
        if self.relu:
            y = F.relu(y)
        return y


class DynamicPolicy(StochasticSuperNetwork):
    INPUT_OBS = 'InObs'
    INPUT_ACT = 'InAct'
    OUTPUT_ACT = 'Out'

    def __init__(self, hidden_size, n_layer_pi, n_layer_act, obs_size, action_size, bias=True, *args, **kwargs):
        """
        Represents a 3 Dimensional Neural fabric, in which each layer, scale position has several identical blocks.
        :param n_layer:
        :param n_block:
        :param n_chan:
        :param data_prop:
        :param kernel_size:
        :param bias:
        :param args:
        :param kwargs:
        """

        super(DynamicPolicy, self).__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.n_layer_pi = n_layer_pi
        self.n_layer_act = n_layer_act
        self.obs_size = obs_size
        self.action_size = action_size

        self.bias = bias

        self.loss = nn.CrossEntropyLoss(reduction='none')

        self.graph.add_node(self.INPUT_OBS, module=DummyBlock())
        self.graph.add_node(self.INPUT_ACT, module=DummyBlock())

        # last_obs_node = self.INPUT_OBS
        # last_size = self.obs_size
        # for i in range(self.n_layer_pi):
        #     cur_node = 'pi_OBS_{}'.format(i)
        #     cur_module = LinearBlock(last_size, self.hidden_size, self.bias, relu=True)
        #
        #     self.graph.add_node(cur_node, module=cur_module)
        #     self.graph.add_edge(last_obs_node, cur_node, width_node=cur_node)
        #
        #     self.blocks.append(cur_module)
        #     self.register_stochastic_node(cur_node)
        #
        #     last_size = self.hidden_size
        #     last_obs_node = cur_node

        last_obs_node = self.stack_layers(self.n_layer_pi, 'pi_OBS_{}', self.INPUT_OBS, self.obs_size)
        last_act_node = self.stack_layers(self.n_layer_act, 'pi_ACT_{}', self.INPUT_ACT, self.action_size)
        # last_act_node = self.INPUT_ACT
        # last_size = self.action_size
        # for i in range(self.n_layer_act):
        #     cur_node = 'pi_ACT_{}'.format(i)
        #     cur_module = LinearBlock(last_size, self.hidden_size, self.bias, relu=True)
        #
        #     self.graph.add_node(cur_node, module=cur_module)
        #     self.graph.add_edge(last_act_node, cur_node, width_node=cur_node)
        #
        #     self.blocks.append(cur_module)
        #     self.register_stochastic_node(cur_node)
        #
        #     last_size = self.hidden_size
        #     last_act_node = cur_node

        cur_module = Output(self.hidden_size, self.action_size, self.bias)

        self.graph.add_node(self.OUTPUT_ACT, module=cur_module)
        self.graph.add_edge(last_obs_node, self.OUTPUT_ACT, width_node=self.OUTPUT_ACT)
        self.graph.add_edge(last_act_node, self.OUTPUT_ACT, width_node=self.OUTPUT_ACT)

        self.blocks.append(cur_module)
        self.register_stochastic_node(self.OUTPUT_ACT)

        self.set_graph(self.graph, [self.INPUT_OBS, self.INPUT_ACT], [self.OUTPUT_ACT])

        self.saved_actions = []
        self.rewards = []

    def stack_layers(self, n, name_format, last_node, last_size):
        for i in range(n):
            cur_node = name_format.format(i)
            cur_module = LinearBlock(last_size, self.hidden_size, self.bias, relu=True)

            self.graph.add_node(cur_node, module=cur_module)
            self.graph.add_edge(last_node, cur_node, width_node=cur_node)

            self.blocks.append(cur_module)
            self.register_stochastic_node(cur_node)

            last_size = self.hidden_size
            last_node = cur_node

        return last_node



if __name__ == '__main__':
    pol = DynamicPolicy(128, 3, 1, 8, 4, deter_eval=True)

    print(pol)
    for e in pol.net.edges:
        print(e)

    obs = torch.rand(1, 8)
    act = torch.zeros(1, 4)

    pol.set_probas(torch.ones(1,5))

    pol.log_probas = []
    y = pol([obs,act])
    print(y)