import jax
import jax.numpy as jnp


def prob_of_binned_vector(x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the probability of each unique binned vector in `x`.
    :param x: A 2D array where each row is a binned vector.
    :return: A 1D array of probabilities corresponding to each unique binned vector.
    """
    assert x.shape[-1] > 0, "Input array must have at least one column."
    assert jnp.all(jnp.array(x.shape) > 0), "Input array must not be empty."
    assert len(x.shape) >= 2, "Input array must be 2D."

    unique_vectors, counts = jnp.unique(x, axis=0, return_counts=True)
    probabilities = counts / jnp.sum(counts)
    return probabilities


def mutual_information_from_binned_vectors(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the mutual information between two arrays of binned vectors. The computation proceeds by first finding all
    unique vectors in `x` and `y`. Then the marginal probabilities are obtained by counting the occurrences of each
    unique vector in `x` and `y`. Finally, the joint probabilities are computed by concatenating the pairs of `x` and `y`
    and counting their occurrences in the same way.
    :param x: 2D array where each row is a binned vector.
    :param y: 2D array where each row is a binned vector.
    :return: Approximate mutual information between `x` and `y`.
    """
    assert (jnp.array(x.shape)[:-1] == jnp.array(y.shape)[:-1]).all(), \
        "x and y must have the same number of rows (samples)."

    # Number of samples
    n = x.shape[0]

    # Map each unique row in x and y to integer ids
    _, x_ids = jnp.unique(x, axis=0, return_inverse=True)
    _, y_ids = jnp.unique(y, axis=0, return_inverse=True)
    nx = int(x_ids.max()) + 1
    ny = int(y_ids.max()) + 1

    # Build joint distribution pxy based on counts via a single bincount on combined ids
    joint_ids = x_ids * ny + y_ids
    joint_counts = jnp.bincount(joint_ids, length=nx * ny)
    pxy = (joint_counts / n).reshape(nx, ny)

    # Marginals from the joint distribution
    px = pxy.sum(axis=1, keepdims=True)       # shape (nx, 1)
    py = pxy.sum(axis=0, keepdims=True)       # shape (1, ny)

    # Compute mutual information
    mask = pxy > 0
    log_term = jnp.log(pxy[mask]) - jnp.log(px @ py)[mask]
    mi_nats = jnp.sum(pxy[mask] * log_term)
    return mi_nats


