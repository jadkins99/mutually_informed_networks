import jax
import numpy as np
import pytest
import jax.numpy as jnp
from binning import bin_into_uniform_bins, bin_into_equal_population_bins


def slow_python_equal_population_binning(data: jnp.ndarray, num_bins: int) -> jnp.ndarray:
    """ Single-column version of the above for easier testing. """
    data = np.array(data)
    bins = np.empty_like(data, dtype=np.int32)

    def bin_col(col: np.array) -> np.array:
        # We perform the binning such that it satisfies two requirements.
        # 1) The bins include roughly equal number of points.
        # 2) Identical values go into the same bin.
        # To achieve point one, we would want to use `equal_bins`. But to achieve point two, we need to
        # round those values to the nearest breakpoint between unique values
        # Unique values and counts
        # Unique values and counts
        values, counts = np.unique(col, return_counts=True)
        n_unique = len(values)

        cum_counts = np.cumsum(counts)
        total = cum_counts[-1]
        cum_probs = cum_counts / total

        # Target cumulative probabilities for bins
        if num_bins <= 1:
            return np.zeros_like(col, dtype=int), np.array([-np.inf, np.inf])

        targets = np.linspace(1 / num_bins, 1 - 1 / num_bins, num_bins - 1)

        # We can only cut after indices 0 .. n_unique-2
        max_cut_idx = n_unique - 2
        cut_indices = []
        prev_idx = -1

        for t in targets:
            start = prev_idx + 1
            if start > max_cut_idx:
                # No more valid cut positions left
                break

            candidate_idxs = np.arange(start, max_cut_idx + 1)
            diffs = np.abs(cum_probs[candidate_idxs] - t)
            best_rel = np.argmin(diffs)
            best_idx = int(candidate_idxs[best_rel])

            cut_indices.append(best_idx)
            prev_idx = best_idx

        cut_indices = np.array(cut_indices, dtype=int)

        # Build bin edges: (-inf, values[cut1], values[cut2], ..., +inf)
        inner_edges = values[cut_indices]
        bin_edges = np.concatenate(([-np.inf], inner_edges, [np.inf]))

        # Assign points to bins
        # np.digitize with right=True → bin i is (edge[i-1], edge[i]]
        bin_indices = np.digitize(col, bin_edges[1:], right=True)

        return bin_indices

    num_cols = data.shape[1]
    for col_idx in range(num_cols):
        column = data[:, col_idx].copy()
        bins[:, col_idx] = bin_col(column)

    return jnp.array(bins)


@pytest.mark.parametrize("binning_function", [slow_python_equal_population_binning, bin_into_uniform_bins, bin_into_equal_population_bins])
def test_unique_value_unique_bins(binning_function):
    data = jnp.array([[0], [1], [2]])
    bins = binning_function(data, num_bins=3)
    assert bins.shape == data.shape, f'Incorrect shape: {bins.shape} for function {binning_function.__name__}'
    assert [[0], [1], [2]] == bins.tolist(), f'Incorrect bins: {bins.tolist()} for function {binning_function.__name__}'


@pytest.mark.parametrize("binning_function", [slow_python_equal_population_binning, bin_into_uniform_bins, bin_into_equal_population_bins])
def test_same_value_same_bins(binning_function):
    data = jnp.array([[0], [0], [1], [2]])
    bins = binning_function(data, num_bins=3)
    assert bins.shape == data.shape, f'Incorrect shape: {bins.shape} for function {binning_function.__name__}'
    assert [[0], [0], [1], [2]] == bins.tolist(), f'Incorrect bins: {bins.tolist()} for function {binning_function.__name__}'


@pytest.mark.parametrize("binning_function", [slow_python_equal_population_binning, bin_into_uniform_bins, bin_into_equal_population_bins])
def test_single_value_single_bin(binning_function):
    data = jnp.array([[0], [0]])
    bins = binning_function(data, num_bins=3)
    assert bins.shape == data.shape, f'Incorrect shape: {bins.shape} for function {binning_function.__name__}'
    assert bins.min() == bins.max(), f'Incorrect bins: {bins.tolist()} for function {binning_function.__name__}'


@pytest.mark.parametrize("binning_function", [slow_python_equal_population_binning, bin_into_uniform_bins, bin_into_equal_population_bins])
def test_different_columns_have_independent_bin_edges(binning_function):
    data = jnp.array([[0, 1],
                      [1, 2],
                      [2, 3]])
    bins = binning_function(data, num_bins=3)
    assert bins.shape == data.shape, f'Incorrect shape: {bins.shape} for function {binning_function.__name__}'
    assert [[0, 0],
            [1, 1],
            [2, 2]] == bins.tolist(), f'Incorrect bins: {bins.tolist()} for function {binning_function.__name__}'


@pytest.mark.parametrize("binning_function", [slow_python_equal_population_binning, bin_into_uniform_bins, bin_into_equal_population_bins])
def test_binning_puts_zeros_into_same_bin(binning_function):
    """ Because the networks may use ReLUs, there might be a lot of zeros which we want to group together. """
    data = jnp.array([[0], [0], [0], [0], [0], [0], [1], [2]])
    bins = binning_function(data, num_bins=3)
    python_bins = slow_python_equal_population_binning(data, num_bins=3)
    assert jnp.array_equal(bins, python_bins), \
        f'JAX bins {bins.tolist()} do not match Python bins {python_bins.tolist()} for function {binning_function.__name__}'
    assert bins.shape == data.shape, f'Incorrect shape: {bins.shape} for function {binning_function.__name__}'
    assert [[0], [0], [0], [0], [0], [0], [1], [2]] == bins.tolist(), \
        f'Incorrect bins: {bins.tolist()} for function {binning_function.__name__}'


@pytest.mark.parametrize("binning_function", [slow_python_equal_population_binning, bin_into_uniform_bins, bin_into_equal_population_bins])
def test_uniform_binning_puts_ones_into_same_bin(binning_function):
    data = jnp.array([[1], [1], [1], [1], [1], [1], [0], [2]])
    bins = binning_function(data, num_bins=3)
    assert bins.shape == data.shape, f'Incorrect shape: {bins.shape} for function {binning_function.__name__}'
    first_bin = bins[0, 0]
    second_bin = bins[-1, 0]
    third_bin = bins[-2, 0]
    assert (jnp.all(bins[:-2, 0] == first_bin) and
            (second_bin != first_bin) and (third_bin != first_bin) and (second_bin != third_bin)), \
        f'Incorrect bins: {bins.tolist()} for function {binning_function.__name__}'


def test_uniform_binning_of_random_data():
    key = jax.random.PRNGKey(0)
    data = jax.random.normal(key, shape=(100, 5))
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


@pytest.mark.parametrize("binning_function", [bin_into_uniform_bins, bin_into_equal_population_bins])
def test_binning_raises_on_no_feature_input(binning_function):
    data = jnp.array([[], [], []])  # shape (3, 0)
    with pytest.raises(AssertionError):
        binning_function(data, num_bins=3)