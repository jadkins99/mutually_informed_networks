import jax
import jax.numpy as jnp

from binned_mutual_information import uniformly_bin_and_compute_mi


def compute_layer_wise_mi_with_uniform_binning(model, x: jnp.ndarray, y: jnp.ndarray, num_bins: int = 30) -> tuple[list[float], list[float]]:
    """
    Compute the layer-wise mutual information between the inputs and each layer's activations,
    as well as between each layer's activations and the outputs.
    :param model: The neural network model.
    :param x: Input data.
    :param y: Output data (labels).
    :param num_bins: Number of bins to use for uniform binning.
    :return: A list of mutual information values for each layer.
    """
    # Make sure we don't affect gradient elsewhere
    model = jax.tree_util.tree_map(jnp.array, model)

    mi_with_input = []
    mi_with_output = []

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
        mi_input_layer = uniformly_bin_and_compute_mi(x, act, num_bins=num_bins)
        mi_with_input.append(mi_input_layer)

    # Compute MI between each layer's activations and output
    for act in activations:
        mi_layer_output = uniformly_bin_and_compute_mi(act, y, num_bins=num_bins)
        mi_with_output.append(mi_layer_output)

    return mi_with_input, mi_with_output