import jax.numpy as jnp


def bin_into_uniform_bins(data: jnp.ndarray, num_bins: int, min_value: float = 0.0, max_value: float = 10.0):
    """
    Bin the input data into uniform bins.
    :param data: Input data to be binned.
    :param num_bins: Number of bins to create.
    :param min_value: Minimum value for binning.
    :param max_value: Maximum value for binning.
    :return: Binned data as integer indices.
    """
    # Compute bin edges
    bin_edges = jnp.linspace(min_value, max_value, num_bins + 1)

    # Digitize the data into bins
    binned_data = jnp.digitize(data, bin_edges) - 1  # Subtract 1 to get 0-based indices

    # Clip values to ensure they fall within valid range
    binned_data = jnp.clip(binned_data, 0, num_bins - 1)

    return binned_data