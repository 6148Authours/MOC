from collections import namedtuple
from typing import List, Callable

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch.autograd import Variable

N_LANDMARKS = 3
N_FOOD = 3
N_FORESTS = 3
N_LAND = 3

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'a_log_prob', 'next_state', 'subgoal'])


def reverse_unroll(params: List[Tensor],
                   hparams: List[Tensor],
                   outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
                   set_grad=True) -> List[Tensor]:
    """
    Computes the hypergradient by backpropagating through a previously employed inner solver procedure.

    Args:
        params: the output of a torch differentiable inner solver (it must depend on hparams in the torch graph)
        hparams: the outer variables (or hyperparameters), each element needs requires_grad=True
        outer_loss: computes the outer objective taking parameters and hyperparameters as inputs
        set_grad: if True set t.grad to the hypergradient for every t in hparams

    Returns:
        the list of hypergradients for each element in hparams
    """
    o_loss = outer_loss(params, hparams)
    grads = torch.autograd.grad(o_loss, hparams, retain_graph=True, create_graph=True, allow_unused=True)
    if set_grad:
        update_tensor_grads(hparams, grads)
    print(grads)
    return grads


def update_tensor_grads(hparams, grads):
    for l, g in zip(hparams, grads):
        if l.grad is None:
            l.grad = torch.zeros_like(l)
        if g is not None:
            l.grad += g


class ReplayBuffer:
    def __init__(self, capacity, num_states):
        self.mem = np.zeros((capacity, num_states * 2 + 2))
        self.memory_counter = 0
        self.capacity = capacity

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.capacity % self.memory_counter
        self.mem[index, :] = transition
        self.memory_counter += 1


def orthogonal(shape):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)


def lstm_ortho_initializer(scale=1.0):
    def _initializer(shape, dtype=torch.float32, partition_info=None):
        size_x = shape[0]
        size_h = shape[1] / 4  # assumes lstm
        t = np.zeros(shape)
        t[:, :size_h] = orthogonal([size_x, size_h]) * scale
        t[:, size_h:size_h * 2] = orthogonal([size_x, size_h]) * scale
        t[:, size_h * 2:size_h * 3] = orthogonal([size_x, size_h]) * scale
        t[:, size_h * 3:] = orthogonal([size_x, size_h]) * scale
        return torch.from_numpy(t).type(dtype)

    return _initializer()


def super_linear(x, output_size, reuse=False, init_w='ortho',
                 weight_start=0.0, use_bias=True, bias_start=0.0, input_size=None):
    shape = x.shape()

    w_init = None
    if input_size is None:
        x_size = shape[1]
    else:
        x_size = input_size
    h_size = output_size
    if init_w == 'zeros':
        w_init = torch.zeros(1)
    elif init_w == 'constant':
        w_init = torch.FloatTensor(weight_start)
    elif init_w == 'gaussian':
        w_init = torch.nn.init.normal_(torch.empty(1), std=weight_start)
    elif init_w == 'ortho':
        w_init = lstm_ortho_initializer(1.0)

    w = w_init.view(x_size, output_size)
    if use_bias:
        b = torch.zeros(output_size)
        return torch.matmul(x, w) + b
    return torch.matmul(x, w)


def hyper_norm(layer, hyper_output, embedding_size, num_units, use_bias=True):
    init_gamma = 0.10
    zw = super_linear(hyper_output, embedding_size, init_w='constant', weight_start=0.0, use_bias=True, bias_start=1.0)
    alpha = super_linear(zw, num_units, init_w='constant', weight_start=init_gamma / embedding_size, use_bias=False)
    result = alpha * layer
    return result


def layer_norm_all(h, batch_size, base, num_units, gamma_start=1.0, epsilon=1e-3, use_bias=True):
    h_reshape = torch.reshape(h, (batch_size, base, num_units))
    mean = h_reshape.mean(2)
    var = torch.square(h_reshape - mean).mean(2)
    epsilon = Variable(epsilon)
    rstd = torch.rsqrt(var + epsilon)
    h_reshape = (h_reshape - mean) * rstd
    # reshape back to original
    h = torch.reshape(h_reshape, (batch_size, base * num_units))
    gamma = torch.from_numpy(np.array([gamma_start] * (4 * num_units)))
    beta = torch.from_numpy(np.array([0.0] * (4 * num_units)))
    if use_bias:
        return gamma * h + beta
    return gamma * h


def layer_norm(x, num_units, gamma_start=1.0, epsilon=1e-3, use_bias=True):
    axes = 1
    mean = torch.mean(x, axes, keepdim=True)
    x_shifted = x - mean
    var = torch.mean(torch.square(x_shifted), axes, keepdim=True)
    inv_std = torch.rsqrt(var + epsilon)
    gamma = torch.FloatTensor([gamma_start] * num_units)
    if use_bias:
        beta = torch.FloatTensor([0.0] * num_units)
    output = gamma * (x_shifted) * inv_std
    if use_bias:
        output = output + beta
    return output


def make_env(scenario_name, arglist, benchmark=False):
    import importlib
    from mpe_local.multiagent.environment import MultiAgentEnv

    module_name = "mpe_local.multiagent.scenarios.{}".format(scenario_name)
    scenario_class = importlib.import_module(module_name).Scenario
    ratio = 1.0 if arglist.map_size == "normal" else 2.0
    scenario = scenario_class(n_good=arglist.n_good, n_adv=arglist.n_adv, n_landmarks=N_LANDMARKS, n_food=N_FOOD,
                              n_forests=N_FORESTS,
                              no_wheel=arglist.no_wheel, sight=arglist.sight, alpha=arglist.alpha, ratio=ratio)
    # scenario = scenario_class(n_good=arglist.n_good, n_adv=arglist.n_adv, n_landmarks=arglist.n_landmarks,
    #                           n_food=arglist.n_food, n_forests=arglist.n_forests, no_wheel=arglist.no_wheel,
    #                           sight=arglist.sight, alpha=arglist.alpha, ratio=ratio)
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, scenario.benchmark_data, export_episode=arglist.save_gif_data,
                            scenario=scenario)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, done_callback=scenario.done, info_callback=scenario.info,
                            export_episode=arglist.save_gif_data, scenario=scenario)
    return env
