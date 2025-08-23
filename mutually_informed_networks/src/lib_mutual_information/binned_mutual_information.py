import jax
import jax.numpy as jnp


def prob_of_binned_vector(x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the probability of each unique binned vector in `x`.
    :param x: A 2D array where each row is a binned vector.
    :return: A 1D array of probabilities corresponding to each unique binned vector.
    """
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
    assert x.shape[-1] == y.shape[-1], "x and y must have the same number of columns (features)."

    # Compute unique vectors and their probabilities
    px = prob_of_binned_vector(x)
    py = prob_of_binned_vector(y)

    # Compute joint distribution
    joint_xy = jnp.concatenate((x, y), axis=-1)
    pxy = prob_of_binned_vector(joint_xy)

    # Compute mutual information
    mi = jnp.sum(pxy * jnp.log(pxy / (px[:, None] * py[None, :] + 1e-10) + 1e-10)).mean()
    return mi


