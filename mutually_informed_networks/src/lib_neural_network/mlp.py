from typing import List

import equinox as eqx
import jax


class MLP(eqx.Module):
    """
    Implements a simple fully connected feedforward neural network.
    The input, hidden, and output layer sizes are configurable.
    """
    layers: list

    def __init__(self, in_size: int, out_size: int, layer_sizes: List[int], key):
        keys = jax.random.split(key, len(layer_sizes) + 1)
        self.layers = []
        # hidden layers
        prev_layer_size = in_size
        for i, layer_size in enumerate(layer_sizes):
            self.layers.append(
                eqx.nn.Linear(prev_layer_size, layer_size, key=keys[i])
            )
            prev_layer_size = layer_size
        # output layer
        self.layers.append(eqx.nn.Linear(prev_layer_size, out_size, key=keys[-1]))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return jax.nn.softmax(self.layers[-1](x))
