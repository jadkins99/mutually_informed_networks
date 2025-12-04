import jax
import pytest
import jax.numpy as jnp
import numpy as np

from binned_mutual_information import mutual_information_from_binned_vectors, uniformly_bin_and_compute_mi, \
    quantile_bin_and_compute_mi
from collections import Counter


def _slow_python_mutual_information(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """A slow, pure Python implementation of mutual information for testing purposes."""
    n_samples = x.shape[0]
    pdf_x, pdf_y, pdf_xy = Counter(), Counter(), Counter()

    for i in range(n_samples):
        x_tuple = tuple(x[i].tolist())
        y_tuple = tuple(y[i].tolist())
        pdf_x[x_tuple] += 1
        pdf_y[y_tuple] += 1
        pdf_xy[(x_tuple, y_tuple)] += 1

    mi = 0.0
    for (x_val, y_val), joint_count in pdf_xy.items():
        p_xy = joint_count / n_samples
        p_x = pdf_x[x_val] / n_samples
        p_y = pdf_y[y_val] / n_samples
        mi += p_xy * np.log(p_xy / (p_x * p_y))

    return mi


def _slow_jax_mutual_information(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the mutual information between two arrays of binned vectors. The computation proceeds by first finding all
    unique vectors in `x` and `y`. Then the marginal probabilities are obtained by counting the occurrences of each
    unique vector in `x` and `y`. Finally, the joint probabilities are computed by concatenating the pairs of `x` and `y`
    and counting their occurrences in the same way.

    This is a legacy slow implementation for testing purposes.
    :param x: 2D array where each row is a binned vector.
    :param y: 2D array where each row is a binned vector.
    :return: Approximate mutual information between `x` and `y`.
    """

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


def test_single_vector_mutual_information():
    x = jnp.array([[0, 1],])
    y = jnp.array([[1, 0],])
    mi = mutual_information_from_binned_vectors(x, y)
    expected_mi = _slow_python_mutual_information(x, y)
    assert expected_mi == 0.0, f"Expected {expected_mi}, got {mi}"  # No uncertainty with single vector
    assert jnp.isclose(mi, expected_mi), f"Expected {expected_mi}, got {mi}"


def test_equal_ys_mutual_information():
    x = jnp.array([[0, 1], [1, 0]])
    y = jnp.array([[1, 0], [1, 0]])
    mi = mutual_information_from_binned_vectors(x, y)
    expected_mi = _slow_python_mutual_information(x, y)
    assert expected_mi == 0.0, f"Expected {expected_mi}, got {mi}"  # No uncertainty with single vector
    assert jnp.isclose(mi, expected_mi), f"Expected {expected_mi}, got {mi}"


def test_identical_vectors_mutual_information():
    x = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mi = mutual_information_from_binned_vectors(x, y)
    # All vectors are identical, se we should get
    # sum_X sum_Y p(x,y) log(p(x,y)/(p(x)p(y))) = sum_X sum_Y log(3) / 3 = log(3)
    expected_mi = _slow_python_mutual_information(x, y)
    assert expected_mi == jnp.log(3)
    assert jnp.isclose(mi, expected_mi), f"Expected {expected_mi}, got {mi}"


def test_different_dimensions_mutual_information():
    x = jnp.array([[0, 1], [1, 0], [0, 1], [1, 1]])
    y = jnp.array([[0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    mi = mutual_information_from_binned_vectors(x, y)
    expected_mi = _slow_python_mutual_information(x, y)
    assert jnp.isclose(mi, expected_mi), f"Expected {expected_mi}, got {mi}"


def test_different_num_unique_mutual_information():
    x = jnp.array([[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    y = jnp.array([[0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    mi = mutual_information_from_binned_vectors(x, y)
    expected_mi = _slow_python_mutual_information(x, y)
    assert jnp.isclose(mi, expected_mi), f"Expected {expected_mi}, got {mi}"


@pytest.mark.parametrize('mi_with_binning_impl', [uniformly_bin_and_compute_mi, quantile_bin_and_compute_mi])
def test_invalid_input_mutual_information(mi_with_binning_impl):
    x = jnp.array([[0, 1], [1, 0]])
    y = jnp.array([[1, 0]])  # Different number of rows
    with pytest.raises(AssertionError):
        mi_with_binning_impl(x, y, num_bins=5)

    y = jnp.array([1, 0])  # Not a 2D array
    with pytest.raises(AssertionError):
        mi_with_binning_impl(x, y, num_bins=5)

    x = jnp.array([[[1, 0]]])
    y = jnp.array([[[1, 0]]])  # 3D array
    with pytest.raises(AssertionError):
        mi_with_binning_impl(x, y, num_bins=5)


def test_mutual_information_non_negative_for_random_data():
    # Seed for reproducibility, but use a different one each time
    seed = jax.random.randint(jax.random.PRNGKey(0), (), 0, 10000)
    print('[test_mutual_information_non_negative_for_random_data]: Using seed', seed)
    for s in range(5):
        key = jax.random.PRNGKey(seed + s)
        x = jax.random.randint(key, (10, 3), 0, 5)
        y = jax.random.randint(key, (10, 3), 0, 5)
        mi = mutual_information_from_binned_vectors(x, y)
        assert mi >= 0, f"Mutual information should be non-negative, got {mi}"


@pytest.mark.parametrize('mi_test_impl', [_slow_python_mutual_information, _slow_jax_mutual_information])
def test_random_data_consistency(mi_test_impl):
    # Seed for reproducibility, but use a different one each time
    seed = jax.random.randint(jax.random.PRNGKey(0), (), 0, 10000)
    print('[test_random_data_consistency]: Using seed', seed)

    key = jax.random.PRNGKey(seed)
    x = jax.random.randint(key, (15, 4), 0, 7)
    y = jax.random.randint(key, (15, 4), 0, 7)
    mi1 = mutual_information_from_binned_vectors(x, y)
    mi2 = mi_test_impl(x, y)
    assert jnp.isclose(mi1, mi2), f"Mutual information should be consistent, got {mi1} and {mi2}"
