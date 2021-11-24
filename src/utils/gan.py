import copy

import torch
import torch.nn as nn
import torch.functional as F
import numpy as np

# Here we use a function to avoid reuse of objects
DEFAULT_GAN_CONFIGS = lambda: {
    'batch_size': 64,
    'generator_output_activation': 'tanh',
    'generator_hidden_activation': 'relu',
    'discriminator_hidden_activation': 'leaky_relu',
    'generator_optimizer': torch.optim.RMSprop,
    'discriminator_optimizer': torch.optim.RMSprop,
    'generator_weight_initializer': nn.init.xavier_normal_,
    'discriminator_weight_initializer': nn.init.xavier_normal_,
    'print_iteration': 50,
    'reset_generator_optimizer': False,
    'reset_discriminator_optimizer': False,
    'batch_normalize_discriminator': False,
    'batch_normalize_generator': False,
    'supress_all_logging': False,
    'default_generator_iters': 1,  # It is highly recommend to not change these parameters
    'default_discriminator_iters': 1,
    'gan_type': 'lsgan',
    'wgan_gradient_penalty': 0.1,
}


class Generator(nn.Module):

    def __init__(self, output_size, hidden_layers, noise_size, configs):
        super(Generator, self).__init__()
        self.configs = configs
        self.layers = nn.ModuleList()
        self.batch_norm = configs['batch_normalize_generator']
        previous_size = noise_size
        for size in hidden_layers:
            self.layers.append(nn.Linear(previous_size, size))
            if configs['generator_hidden_actiovation'] == 'relu':
                self.layers.append(nn.ReLU())
            elif configs['generator_hidden_actiovation'] == 'leaky_relu':
                self.layers.append(nn.LeakyReLU(0.2))
            else:
                raise ValueError('Unsupported activation type')
            previous_size = size

        self.layers.append(nn.Linear(previous_size, output_size))
        if configs['generator_hidden_actiovation'] == 'tanh':
            self.layers.append(nn.Tanh())
        elif configs['generator_hidden_actiovation'] == 'sigmoid':
            self.layers.append(nn.Sigmoid())
        else:
            raise ValueError('Unsupported activation type!')

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if self.batch_norm:
                x = nn.BatchNorm1d(x)
        return x


class DiscriminatorNet(nn.Module):

    def __init__(self, input_size, hidden_layers, output_size, configs):
        super(DiscriminatorNet, self).__init__()

        previous_size = input_size
        self.layers = nn.ModuleList()

        for size in hidden_layers:
            self.layers.append(nn.Linear(previous_size, size))
            if configs['generator_hidden_actiovation'] == 'relu':
                self.layers.append(nn.ReLU())
            elif configs['generator_hidden_actiovation'] == 'leaky_relu':
                self.layers.append(nn.LeakyReLU(0.2))
            else:
                raise ValueError('Unsupported activation type')
            previous_size = size
        self.layers.append(nn.Linear(previous_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if self.batch_norm:
                x = nn.BatchNorm1d(x)
        return x


class Discriminator(nn.Module):

    def __init__(self, generator_output, input_size, hidden_layers, output_size, configs):
        super(Discriminator, self).__init__()
        self._generator_input = generator_output
        self.configs = configs

        self.sample_discriminator = DiscriminatorNet(input_size, hidden_layers, output_size, configs)
        self.generator_discriminator = DiscriminatorNet(input_size, hidden_layers, output_size, configs)

    def forward(self, x):
        return self.sample_discriminator(x), self.generator_discriminator(x)


def batch_feed_array(array, batch_size):
    data_size = array.shape[0]

    if data_size <= batch_size:
        while True:
            yield array

    else:
        start = 0
        while True:
            if start + batch_size < data_size:
                yield array

class FCGAN(object):
    def __init__(self, generator_output_size, discriminator_output_size, generator_layers, discriminator_layers,
                 noise_size, configs=None):
        self.generator_output_size = generator_output_size
        self.discriminator_output_size = discriminator_output_size
        self.noise_size = noise_size
        self.configs = copy.deepcopy(DEFAULT_GAN_CONFIGS)
        if configs is not None:
            self.configs.update(configs)

        self.generator = Generator(generator_output_size, generator_layers, noise_size, self.configs)
        self.discriminator = Discriminator(self.generator.output, generator_output_size, discriminator_layers,
                                           discriminator_output_size, self.configs)
        self.generator_train_op = self.configs['generator_optimizer'](self.generator.parameters())
        self.discriminator_train_op = self.configs['discriminator_optimizer'](self.discriminator.parameters())

    def sample_random_noise(self, size):
        return np.random.randn(size, self.noise_size)

    def sample_generator(self, size):
        generator_samples = []
        generator_noise = []
        batch_size = self.configs['batch_size']
        for i in range(0, size, batch_size):
            sample_size = min(batch_size, size - i)
            noise = self.sample_random_noise(sample_size)
            generator_noise.append(noise)
            # Todo: this part is incorrect
            generator_samples.append(self.generator.output, self.generator.input)

        return np.vstack(generator_samples), np.vstack(generator_noise)

    def train(self, X, Y, outer_iers, generator_iters=None, discriminator_iters=None):
        if generator_iters is None:
            generator_iters = self.configs['default_generator_iters']
        if discriminator_iters is None:
            discriminator_iters = self.configs['default_discriminator_iters']

        sample_size = X.shape[0]
        train_size = sample_size

        batch_size = self.configs['batch_size']

        generated_Y = np.zeros((batch_size, self.discriminator_output_size))

        batch_feed_X = batch
