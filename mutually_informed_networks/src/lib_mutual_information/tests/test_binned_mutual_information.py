import jax
import pytest
import jax.numpy as jnp
import numpy as np

from binned_mutual_information import mutual_information_from_binned_vectors
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


def test_invalid_input_mutual_information():
    x = jnp.array([[0, 1], [1, 0]])
    y = jnp.array([[1, 0]])  # Different number of rows
    with pytest.raises(AssertionError):
        mutual_information_from_binned_vectors(x, y)

    y = jnp.array([1, 0])  # Not a 2D array
    with pytest.raises(AssertionError):
        mutual_information_from_binned_vectors(x, y)


def test_extra_dimensions_mutual_information():
    x = jnp.array([[[0, 1]], [[1, 0]], [[0, 1]], [[1, 1]]])
    y = jnp.array([[[1, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    mi = mutual_information_from_binned_vectors(x, y)
    for d in range(2, 5):
        x_d = jnp.repeat(x, d, axis=1)
        y_d = jnp.repeat(y, d, axis=1)
        multidim_mi = mutual_information_from_binned_vectors(x_d, y_d)
        assert jnp.isclose(multidim_mi, mi), \
            f"Failed for dimension {d}. Expected {mi}, got {multidim_mi}"


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


def test_random_data_consistency():
    # Seed for reproducibility, but use a different one each time
    seed = jax.random.randint(jax.random.PRNGKey(0), (), 0, 10000)
    print('[test_random_data_consistency]: Using seed', seed)

    key = jax.random.PRNGKey(seed)
    x = jax.random.randint(key, (15, 4), 0, 7)
    y = jax.random.randint(key, (15, 4), 0, 7)
    mi1 = mutual_information_from_binned_vectors(x, y)
    mi2 = _slow_python_mutual_information(x, y)
    assert jnp.isclose(mi1, mi2), f"Mutual information should be consistent, got {mi1} and {mi2}"
