import jax
import jax.numpy as jnp

from binned_mutual_information import mutual_information_from_binned_vectors


def compute_layer_wise_mi(model, x: jnp.ndarray, y: jnp.ndarray) -> tuple[list[float], list[float]]:
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

    # Function to extract activations from each layer
    def _get_activations(model, x):
        activations = []
        current_input = x
        for layer in model.layers:
            current_input = layer(current_input)
            activations.append(current_input)
        return activations

    # Get activations for all layers
    activations = _get_activations(model, x)

    # Compute MI between input and each layer's activations
    for act in activations:
        mi_input_layer = mutual_information_from_binned_vectors(x.astype(jnp.int32), act.astype(jnp.int32))
        mi_with_input.append(mi_input_layer)

    # Compute MI between each layer's activations and output
    for act in activations:
        mi_layer_output = mutual_information_from_binned_vectors(act.astype(jnp.int32), y.astype(jnp.int32))
        mi_with_output.append(mi_layer_output)

    return mi_with_input, mi_with_output