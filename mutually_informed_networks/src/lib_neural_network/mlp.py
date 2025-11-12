import equinox as eqx
import jax


class MLP(eqx.Module):
    """
    Implements a simple fully connected feedforward neural network.
    The input, hidden, and output layer sizes are configurable.
    """
    layers: list

    def __init__(self, in_size, out_size, width_size, depth, key):
        keys = jax.random.split(key, depth + 1)
        self.layers = []
        # hidden layers
        for i in range(depth):
            self.layers.append(
                eqx.nn.Linear(
                    in_size if i == 0 else width_size,
                    width_size,
                    key=keys[i]
                )
            )
        # output layer
        self.layers.append(eqx.nn.Linear(width_size, out_size, key=keys[-1]))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return jax.nn.softmax(self.layers[-1](x))
