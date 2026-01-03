from copy import deepcopy

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from layer_wise_mi import compute_layer_wise_mi_with_quantile_binning


class DropNeuronResetHeuristicCallback:
    """
    A callback class that resets a neuron based on the drop in MI when that neuron is removed.
    We ensure only one neuron is reset per layer per call.
    """
    def __init__(self, num_bins: int, reset_threshold: float):
        self.num_bins = num_bins
        self.reset_threshold = reset_threshold
        self.mean_mi_drop_per_neuron = {}

    def __call__(self, model, x: jnp.ndarray, y: jnp.ndarray):
        neurons_to_reset = []

        _, mi_with_output_full = compute_layer_wise_mi_with_quantile_binning(model, x, y, num_bins=self.num_bins)

        for layer_idx in range(len(model.layers) - 1):
            for neuron_idx in range(model.layers[layer_idx].weight.shape[0]):
                key = f'Layer {layer_idx} Neuron {neuron_idx}'

                _, mi_with_output = mutual_information_without_neuron(
                    model, layer_idx, neuron_idx, x, y, num_bins=self.num_bins)

                mi_drop = max(mi_with_output_full[layer_idx] - mi_with_output[layer_idx], 1e-3)
                max_layer_drop = np.max([self.mean_mi_drop_per_neuron.get(f'Layer {layer_idx} Neuron {n}', 1e-3) for n in range(model.layers[layer_idx].weight.shape[0])])
                mi_drop_relative_to_last = mi_drop / max_layer_drop

                if mi_drop_relative_to_last < self.reset_threshold:
                    neurons_to_reset.append((layer_idx, neuron_idx))
                    try:
                        del self.mean_mi_drop_per_neuron[key]
                    except:
                        pass
                else:
                    alpha = 0.5
                    self.mean_mi_drop_per_neuron[key] = alpha * self.mean_mi_drop_per_neuron.get(key, mi_drop) + (1 - alpha) * mi_drop
                    # self.mean_mi_drop_per_neuron[key] = max(self.mean_mi_drop_per_neuron.get(key, mi_drop), mi_drop)

        for layer_idx, lowers_neuron_idx in neurons_to_reset:
            print(f'Resetting neuron {lowers_neuron_idx} in layer {layer_idx}.')
            model = _reset_neuron_in_mlp(model, layer_idx, lowers_neuron_idx)

        return model


def _reset_neuron_in_mlp(model, layer_idx, neuron_idx):
    """Resets a neuron in an MLP model by reinitializing its weights and biases."""
    new_layers = []
    _, model_key = jrandom.split(jrandom.PRNGKey(seed=42), 2)
    for idx, layer in enumerate(model.layers):
        if idx == layer_idx:
            # Reinitialize the weights and biases of the specified neuron
            new_layer = eqx.nn.Linear(layer.weight.shape[1], layer.weight.shape[0], key=model_key)
            new_weights = new_layer.weight.at[neuron_idx, :].set(
                jrandom.normal(model_key, (layer.weight.shape[1],)) * jnp.sqrt(2 / layer.weight.shape[1])
            )
            new_bias = new_layer.bias.at[neuron_idx].set(0.0)
            new_layer = eqx.tree_at(lambda l: l.weight, new_layer, new_weights)
            new_layer = eqx.tree_at(lambda l: l.bias, new_layer, new_bias)
            new_layers.append(new_layer)
        else:
            new_layers.append(layer)

    # Create a copy of the input model and replace its layers with the modified ones
    new_model = deepcopy(model)
    new_model = eqx.tree_at(lambda m: m.layers, new_model, new_layers)
    return new_model


def _remove_neuron_from_mlp(model, layer_idx, neuron_idx):
    """Removes a neuron from an MLP model by zeroing its weights and biases."""
    new_layers = []
    _, model_key = jrandom.split(jrandom.PRNGKey(seed=42), 2)
    for idx, layer in enumerate(model.layers):
        if idx == layer_idx:
            # Zero out the weights and biases of the specified neuron
            new_weights = layer.weight.at[neuron_idx, :].set(0)
            new_bias = layer.bias.at[neuron_idx].set(0)
            new_layer = eqx.nn.Linear(new_weights.shape[1], new_weights.shape[0], key=model_key)
            new_layer = eqx.tree_at(lambda l: l.weight, new_layer, new_weights)
            new_layer = eqx.tree_at(lambda l: l.bias, new_layer, new_bias)
            new_layers.append(new_layer)
        else:
            new_layers.append(layer)

    # Create a copy of the input model and replace its layers with the modified ones
    new_model = deepcopy(model)
    new_model = eqx.tree_at(lambda m: m.layers, new_model, new_layers)
    return new_model


def mutual_information_without_neuron(
    model,
    layer_idx: int,
    neuron_idx: int,
    x: jnp.ndarray,
    y: jnp.ndarray,
    num_bins: int,
):
    """
    Computes the mutual information of the model after removing a specific neuron
    from a specified layer.

    :param model: The original MLP model.
    :param layer_idx: The index of the layer from which to remove the neuron.
    :param neuron_idx: The index of the neuron to remove.
    :param x: Input data.
    :param y: Output data.
    :param num_bins: Number of bins for quantile binning in MI computation.
    :return: Mutual information with input and output after neuron removal.
    """
    modified_model = _remove_neuron_from_mlp(model, layer_idx, neuron_idx)

    mi_with_input, mi_with_output = compute_layer_wise_mi_with_quantile_binning(
        modified_model, x, y, num_bins=num_bins
    )

    return mi_with_input, mi_with_output
