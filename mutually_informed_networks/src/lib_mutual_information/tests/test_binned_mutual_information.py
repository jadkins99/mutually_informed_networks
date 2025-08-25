import jax
import pytest
import jax.numpy as jnp

from binned_mutual_information import prob_of_binned_vector, mutual_information_from_binned_vectors


def test_prob_of_single_binned_vector():
    x = jnp.array([[0, 1],])
    probabilities = prob_of_binned_vector(x)
    expected_probabilities = jnp.array([1.0])  # Only one unique vector
    assert jnp.allclose(probabilities, expected_probabilities), \
        f"Expected {expected_probabilities}, got {probabilities}"


def test_prob_of_single_unique_binned_vector():
    x = jnp.array([[0, 1], [0, 1], [0, 1]])
    probabilities = prob_of_binned_vector(x)
    expected_probabilities = jnp.array([1.0])  # Only one unique vector
    assert jnp.allclose(probabilities, expected_probabilities), \
        f"Expected {expected_probabilities}, got {probabilities}"


def test_prob_of_multiple_binned_vectors():
    x = jnp.array([[0, 1], [1, 0], [0, 1], [1, 1]])
    probabilities = prob_of_binned_vector(x)
    expected_probabilities = jnp.array([0.5, 0.25, 0.25])  # Three unique vectors
    assert jnp.allclose(jnp.sort(probabilities), jnp.sort(expected_probabilities)), \
        f"Expected {expected_probabilities}, got {probabilities}"


def test_extra_dimensions_binned_vectors():
    x = jnp.array([[[0, 1]], [[1, 0]], [[0, 1]], [[1, 1]]])
    probabilities = prob_of_binned_vector(x)
    for d in range(2, 5):
        y = jnp.repeat(x, d, axis=1)
        multidim_probabilities = prob_of_binned_vector(y)
        assert jnp.allclose(multidim_probabilities, probabilities), \
            f"Failed for dimension {d}. Expected {probabilities}, got {multidim_probabilities}"


def test_empty_input_binned_vectors():
    x = jnp.empty((0, 2))
    with pytest.raises(AssertionError):
        prob_of_binned_vector(x)

    x = jnp.empty((3, 0))
    with pytest.raises(AssertionError):
        prob_of_binned_vector(x)


def test_invalid_input_binned_vectors():
    x = jnp.array([1, 2, 3])  # Not a 2D array
    with pytest.raises(AssertionError):
        prob_of_binned_vector(x)


def test_single_vector_mutual_information():
    x = jnp.array([[0, 1],])
    y = jnp.array([[1, 0],])
    mi = mutual_information_from_binned_vectors(x, y)
    expected_mi = 0.0  # No uncertainty with single vector
    assert jnp.isclose(mi, expected_mi), f"Expected {expected_mi}, got {mi}"


def test_equal_ys_mutual_information():
    x = jnp.array([[0, 1], [1, 0]])
    y = jnp.array([[1, 0], [1, 0]])
    mi = mutual_information_from_binned_vectors(x, y)
    expected_mi = 0.0  # No uncertainty in y
    assert jnp.isclose(mi, expected_mi), f"Expected {expected_mi}, got {mi}"


def test_identical_vectors_mutual_information():
    x = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mi = mutual_information_from_binned_vectors(x, y)
    # All vectors are identical, se we should get
    # sum_X sum_Y p(x,y) log(p(x,y)/(p(x)p(y))) = sum_X sum_Y log(3) / 3 = log(3) * 3
    expected_mi = jnp.log(3) * 3
    assert jnp.isclose(mi, expected_mi), f"Expected {expected_mi}, got {mi}"


def test_different_dimensions_mutual_information():
    x = jnp.array([[0, 1], [1, 0], [0, 1], [1, 1]])
    y = jnp.array([[0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    mi = mutual_information_from_binned_vectors(x, y)
    expected_mi = mi  # The last coordinate of y is redundant
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
