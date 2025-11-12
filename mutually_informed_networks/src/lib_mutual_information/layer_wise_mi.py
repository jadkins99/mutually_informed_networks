import jax
import jax.numpy as jnp

from binned_mutual_information import mutual_information_from_binned_vectors
from binning import bin_into_uniform_bins


def compute_layer_wise_mi(model, x: jnp.ndarray, y: jnp.ndarray, num_bins: int = 30) -> tuple[list[float], list[float]]:
    """
    Compute the layer-wise mutual information between the inputs and each layer's activations,
    as well as between each layer's activations and the outputs.
    :param model: The neural network model.
    :param x: Input data.
    :param y: Output data (labels).
    :return: A list of mutual information values for each layer.
    """
    # Make sure we don't affect gradient elsewhere
    model = jax.tree_util.tree_map(jnp.array, model)

    mi_with_input = []
    mi_with_output = []

    binned_x = bin_into_uniform_bins(x, num_bins=num_bins)
    binned_y = bin_into_uniform_bins(y, num_bins=num_bins)

    # Function to extract activations from each layer
    def _get_activations(model, x):
        activations = []
        current_input = x
        for layer in model.layers:
            current_input = jax.vmap(layer)(current_input)
            activations.append(current_input)
        return activations

    # Get activations for all layers
    activations = _get_activations(model, x)

    # Compute MI between input and each layer's activations
    for act in activations:
        binned_act = bin_into_uniform_bins(act, num_bins=num_bins)
        mi_input_layer = mutual_information_from_binned_vectors(binned_x, binned_act)
        mi_with_input.append(mi_input_layer)

    # Compute MI between each layer's activations and output
    for act in activations:
        binned_act = bin_into_uniform_bins(act, num_bins=num_bins)
        mi_layer_output = mutual_information_from_binned_vectors(binned_act, binned_y)
        mi_with_output.append(mi_layer_output)

    return mi_with_input, mi_with_output