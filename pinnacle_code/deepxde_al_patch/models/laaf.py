""" BASED ON https://github.com/lululxvi/deepxde/blob/master/deepxde/nn/jax/fnn.py """


from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import linen as nn

from .. import deepxde as dde


class FNNWithLAAF(dde.nn.NN):
    """Fully-connected neural network."""

    layer_sizes: Any
    activation: Any
    kernel_initializer: Any
    n_factor: int = 1.

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
            self._activation = list(map(dde.nn.activations.get, self.activation))
        else:
            self._activation = dde.nn.activations.get(self.activation)
        kernel_initializer = dde.nn.initializers.get(self.kernel_initializer)
        initializer = jax.nn.initializers.zeros

        self.denses = [
            nn.Dense(
                unit,
                kernel_init=kernel_initializer,
                bias_init=initializer,
            )
            for unit in self.layer_sizes[1:]
        ]

    @nn.compact
    def __call__(self, inputs, training=False):
        
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
            
        for j, linear in enumerate(self.denses[:-1]):
            
            x = linear(x)
            
            scale = self.param(f'scale_{j}', nn.initializers.constant(1. / self.n_factor), ())
            x = self.n_factor * scale * x
            
            x = (
                self._activation[j](x)
                if isinstance(self._activation, list)
                else self._activation(x)
            )
            
        x = self.denses[-1](x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
            
        return x
    
class FNNWithGAAF(dde.nn.NN):
    """Fully-connected neural network."""

    layer_sizes: Any
    activation: Any
    kernel_initializer: Any
    n_factor: int = 1.

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
            self._activation = list(map(dde.nn.activations.get, self.activation))
        else:
            self._activation = dde.nn.activations.get(self.activation)
        kernel_initializer = dde.nn.initializers.get(self.kernel_initializer)
        initializer = jax.nn.initializers.zeros

        self.denses = [
            nn.Dense(
                unit,
                kernel_init=kernel_initializer,
                bias_init=initializer,
            )
            for unit in self.layer_sizes[1:]
        ]

    @nn.compact
    def __call__(self, inputs, training=False):
        
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)
            
        scale = self.param(f'scale', nn.initializers.constant(1. / self.n_factor), ())

        for j, linear in enumerate(self.denses[:-1]):
            
            x = linear(x)
            x = self.n_factor * scale * x
            
            x = (
                self._activation[j](x)
                if isinstance(self._activation, list)
                else self._activation(x)
            )
            
        x = self.denses[-1](x)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
            
        return x