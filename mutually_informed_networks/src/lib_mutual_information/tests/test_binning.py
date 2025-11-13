import jax
import pytest
import jax.numpy as jnp
from binning import bin_into_uniform_bins, bin_into_equal_population_bins



@pytest.mark.parametrize("binning_function", [bin_into_uniform_bins, bin_into_equal_population_bins])
def test_unique_value_unique_bins(binning_function):
    data = jnp.array([[0], [1], [2]])
    bins = binning_function(data, num_bins=3)
    assert bins.shape == data.shape, f'Incorrect shape: {bins.shape} for function {binning_function.__name__}'
    assert [[0], [1], [2]] == bins.tolist(), f'Incorrect bins: {bins.tolist()} for function {binning_function.__name__}'


@pytest.mark.parametrize("binning_function", [bin_into_uniform_bins, bin_into_equal_population_bins])
def test_same_value_same_bins(binning_function):
    data = jnp.array([[0], [0], [1], [2]])
    bins = binning_function(data, num_bins=3)
    assert bins.shape == data.shape, f'Incorrect shape: {bins.shape} for function {binning_function.__name__}'
    assert [[0], [0], [1], [2]] == bins.tolist(), f'Incorrect bins: {bins.tolist()} for function {binning_function.__name__}'


@pytest.mark.parametrize("binning_function", [bin_into_uniform_bins, bin_into_equal_population_bins])
def test_single_value_single_bin(binning_function):
    data = jnp.array([[0], [0]])
    bins = binning_function(data, num_bins=3)
    assert bins.shape == data.shape, f'Incorrect shape: {bins.shape} for function {binning_function.__name__}'
    assert bins.min() == bins.max(), f'Incorrect bins: {bins.tolist()} for function {binning_function.__name__}'


@pytest.mark.parametrize("binning_function", [bin_into_uniform_bins, bin_into_equal_population_bins])
def test_different_columns_have_independent_bin_edges(binning_function):
    data = jnp.array([[0, 1],
                      [1, 2],
                      [2, 3]])
    bins = binning_function(data, num_bins=3)
    assert bins.shape == data.shape, f'Incorrect shape: {bins.shape} for function {binning_function.__name__}'
    assert [[0, 0],
            [1, 1],
            [2, 2]] == bins.tolist(), f'Incorrect bins: {bins.tolist()} for function {binning_function.__name__}'


@pytest.mark.parametrize("binning_function", [bin_into_uniform_bins, bin_into_equal_population_bins])
def test_binning_puts_zeros_into_same_bin(binning_function):
    """ Because the networks may use ReLUs, there might be a lot of zeros which we want to group together. """
    data = jnp.array([[0], [0], [0], [0], [0], [0], [1], [2]])
    bins = binning_function(data, num_bins=3)
    assert bins.shape == data.shape, f'Incorrect shape: {bins.shape} for function {binning_function.__name__}'
    assert [[0], [0], [0], [0], [0], [0], [1], [2]] == bins.tolist(), \
        f'Incorrect bins: {bins.tolist()} for function {binning_function.__name__}'


def test_uniform_binning_of_random_data():
    key = jax.random.PRNGKey(0)
    data = jax.random.normal(key, shape=(1000, 5))
    num_bins = 10
    bins = bin_into_equal_population_bins(data, num_bins=num_bins)
    assert bins.shape == data.shape, f'Incorrect shape: {bins.shape}'
    assert bins.min() >= 0 and bins.max() < num_bins, f'Bins out of range: min {bins.min()}, max {bins.max()}'

    unique, count = jnp.unique(bins, return_counts=True)
    assert len(unique) == num_bins, f'Not all bins used: found {len(unique)} unique bins'
    assert (count.max() - count.min()) / count.mean() < 0.01, f'Uneven bin populations: counts {count.tolist()}'


@pytest.mark.parametrize("binning_function", [bin_into_uniform_bins, bin_into_equal_population_bins])
def test_binning_raises_on_one_dimensional_input(binning_function):
    data = jnp.array([0, 1, 2])
    with pytest.raises(AssertionError):
        binning_function(data, num_bins=3)