import jax
import jax.numpy as jnp

from binning import bin_into_uniform_bins, bin_into_equal_population_bins


def _assert_valid_input_dims(x: jnp.ndarray, y: jnp.ndarray):
    assert x.shape[-1] > 0, "Input array must have at least one column."
    assert y.shape[-1] > 0, "Target array must have at least one column."
    assert jnp.all(jnp.array(x.shape) > 0), "Input array must not be empty."
    assert jnp.all(jnp.array(y.shape) > 0), "Target array must not be empty."
    assert len(x.shape) == 2, "Input array must be 2D."
    assert len(y.shape) == 2, "Target array must be 2D."
    assert (jnp.array(x.shape)[:-1] == jnp.array(y.shape)[:-1]).all(), \
        "x and y must have the same number of rows (samples)."


def uniformly_bin_and_compute_mi(x: jnp.ndarray, y: jnp.ndarray, num_bins: int) -> jnp.ndarray:
    """
    Uniformly bin the input arrays `x` and `y` into `num_bins` bins and compute the mutual information between them.
    :param x: Input array.
    :param y: Target array.
    :param num_bins: Number of bins to use for uniform binning.
    :return: Approximate mutual information between binned `x` and `y`.
    """
    _assert_valid_input_dims(x, y)
    binned_x = bin_into_uniform_bins(x, num_bins=num_bins)
    binned_y = bin_into_uniform_bins(y, num_bins=num_bins)
    mi = mutual_information_from_binned_vectors(binned_x, binned_y)
    return mi


def quantile_bin_and_compute_mi(x: jnp.ndarray, y: jnp.ndarray, num_bins: int) -> jnp.ndarray:
    """
    Bin the input arrays `x` and `y` into `num_bins` bins with approximately equal population and compute the mutual
    information between them.
    :param x: Input array.
    :param y: Target array.
    :param num_bins: Number of bins to use for equal population binning.
    :return: Approximate mutual information between binned `x` and `y`.
    """
    _assert_valid_input_dims(x, y)
    binned_x = bin_into_equal_population_bins(x, num_bins=num_bins)
    binned_y = bin_into_equal_population_bins(y, num_bins=num_bins)
    mi = mutual_information_from_binned_vectors(binned_x, binned_y)
    return mi


def _row_ids(data: jnp.ndarray) -> jnp.ndarray:
    """
    Given data of shape (n, d), return for each row an integer id in [0, n-1]
    such that identical rows get the same id. Uses only O(n) memory and
    static shapes, so it's JIT-friendly.
    """
    n = data.shape[0]

    # Lexicographic sort by columns
    # jnp.lexsort expects a sequence of keys, last key is primary
    perm = jnp.lexsort(tuple(data.T))        # (n,)
    sorted_data = data[perm]                 # (n, d)

    # Detect boundaries between different rows
    first = jnp.array([True])
    rest = jnp.any(sorted_data[1:] != sorted_data[:-1], axis=1)
    is_new = jnp.concatenate([first, rest])  # (n,)

    # Cumulative sum → unique group id for each run
    group_ids_sorted = jnp.cumsum(is_new.astype(jnp.int32)) - 1  # 0..k-1 (k <= n)

    # Undo the permutation to get ids in original order
    inv_perm = jnp.zeros_like(perm)
    inv_perm = inv_perm.at[perm].set(jnp.arange(n))
    ids = group_ids_sorted[inv_perm]         # (n,)

    return ids


@jax.jit
def mutual_information_from_binned_vectors(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    JIT-friendly MI between arrays of binned vectors x and y.
    x, y: shape (n, dx), (n, dy), entries are (integer) bin labels.
    """
    n = x.shape[0]
    assert y.shape[0] == n

    # Map each distinct row to an id in [0, n-1]
    x_ids = _row_ids(x)                          # (n,)
    y_ids = _row_ids(y)                          # (n,)

    # For the joint, just concatenate the vectors
    joint = jnp.concatenate([x, y], axis=1)      # (n, dx+dy)
    joint_ids = _row_ids(joint)                  # (n,)

    # Counts for each id (length n, many entries will be zero)
    counts_x = jnp.bincount(x_ids, length=n)     # (n,)
    counts_y = jnp.bincount(y_ids, length=n)     # (n,)
    counts_xy = jnp.bincount(joint_ids, length=n)  # (n,)

    # Empirical probabilities for each sample
    px  = counts_x[x_ids]   / n                  # (n,)
    py  = counts_y[y_ids]   / n                  # (n,)
    pxy = counts_xy[joint_ids] / n               # (n,)

    # MI = E[ log pxy - log px - log py ]
    eps = 1e-12
    log_ratio = jnp.log(pxy + eps) - jnp.log(px * py + eps)
    mi_nats = jnp.mean(log_ratio)
    return mi_nats


