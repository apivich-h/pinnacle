from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import linen as nn

from .nn import NN
from .. import activations
from .. import initializers

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module


class PFNN(NN):
    """Fully-connected neural network."""

    layer_sizes: Any
    activation: Any
    kernel_initializer: Any

    params: Any = None
    _input_transform: Callable = None
    _output_transform: Callable = None

    def setup(self):
        # TODO: implement get regularizer
        if isinstance(self.activation, list):
            if not (len(self.layer_sizes) - 1) == len(self.activation):
                raise ValueError(
                    "Total number of activation functions do not match with sum of hidden layers and output layer!"
                )
            self._activation = list(map(activations.get, self.activation))
        else:
            self._activation = activations.get(self.activation)
        kernel_initializer = initializers.get(self.kernel_initializer)
        initializer = jax.nn.initializers.zeros
        
        self.denses = [
            [
                nn.Dense(
                    u,
                    kernel_init=kernel_initializer,
                    bias_init=initializer,
                ) 
                for u in [x[i] for x in self.layer_sizes[1:-1]] + [1]
            ]
            for i in range(self.layer_sizes[-1])
        ]

    def __call__(self, inputs, training=False):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
            
        x_list = []
        for i in range(self.layer_sizes[-1]):
            denses = self.denses[i]
            for j, linear in enumerate(denses[:-1]):
                x = (
                    self._activation[j](linear(x))
                    if isinstance(self._activation, list)
                    else self._activation(linear(x))
                )
            x = denses[-1](x)
            assert x.shape[-1] == 1
            x_list.append(x)
            
        x = jnp.concatenate(x_list, axis=-1)
        
        if self._output_transform is not None:
            ii = len(x.shape)
            if ii == 1:
                x.reshape(1, -1)
                inputs.reshape(1, -1)
            x = self._output_transform(inputs, x)
            if ii == 1:
                x.reshape(-1)
        return x
