from typing import Any, Callable

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn

from .. import deepxde as dde
# from deepxde.nn.jax.nn import NN
# from deepxde.nn import activations, initializers


def generate_fourier_fnn(layer_sizes, activation, kernel_initializer,
                         ff_count=100, W_scale=1.):
    
    default_dtype = jnp.array([1.]).dtype  # get jax default float type
    ff_W = W_scale * jnp.array(np.random.randn(layer_sizes[0], ff_count), dtype=default_dtype)
    ff_b = 2. * jnp.pi * jnp.array(np.random.randn(1, ff_count), dtype=default_dtype)
    
    def _input_transform(x):
        # x = (x @ ff_W + ff_b)
        x = jax.lax.dot_general(
            jnp.array(x, dtype=default_dtype),
            ff_W,
            (((x.ndim - 1,), (0,)), ((), ())),
        )
        x += jnp.reshape(ff_b, (1,) * (x.ndim - 1) + (-1,))
        x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=x.ndim - 1) / (ff_count ** 0.5)
        return x
    
    return dde.nn.FNN(
        layer_sizes=[ff_count] + layer_sizes[1:], 
        activation=activation,
        kernel_initializer=kernel_initializer,
        _input_transform=_input_transform,
    ), (ff_W, ff_b)


# class FourierFNN(NN):
#     """Fully-connected neural network."""

#     layer_sizes: Any
#     activation: Any
#     kernel_initializer: Any
#     ff_count: int = 100
#     W_scale: float = 1.

#     params: Any = None
#     _input_transform: Callable = None
#     _output_transform: Callable = None
    

#     def setup(self):
        
#         self.ff_W = self.W_scale * jnp.array(np.random.randn(self.layer_sizes[0], self.ff_count))
#         self.ff_b = 2. * jnp.pi * jnp.array(np.random.randn(1, self.ff_count))
        
#         # TODO: implement get regularizer
#         if isinstance(self.activation, list):
#             if not (len(self.layer_sizes) - 1) == len(self.activation):
#                 raise ValueError(
#                     "Total number of activation functions do not match with sum of hidden layers and output layer!"
#                 )
#             self._activation = list(map(activations.get, self.activation))
#         else:
#             self._activation = activations.get(self.activation)
#         kernel_initializer = initializers.get(self.kernel_initializer)
#         initializer = jax.nn.initializers.zeros

#         self.denses = [
            # nn.Dense(
            #     unit,
            #     kernel_init=kernel_initializer,
            #     bias_init=initializer,
            # )
#             for unit in self.layer_sizes[1:]
#         ]


#     def __call__(self, inputs, training=False):
#         if self._input_transform is not None:
#             x = self._input_transform(x)
            
#         # RFF part
#         x = (inputs @ self.ff_W + self.ff_b)
#         x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=1) / (self.ff_count ** 0.5)
        
#         for j, linear in enumerate(self.denses[:-1]):
#             x = (
#                 self._activation[j](linear(x))
#                 if isinstance(self._activation, list)
#                 else self._activation(linear(x))
#             )
#         x = self.denses[-1](x)
        
#         if self._output_transform is not None:
#             x = self._output_transform(inputs, x)
#         return x
