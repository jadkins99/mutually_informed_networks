import jax
import jax.numpy as jnp
from jax import lax


def bin_into_uniform_bins(data: jnp.ndarray, num_bins: int):
    """
    Bin the input data into uniform bins.
    :param data: Input data to be binned.
    :param num_bins: Number of bins to create.
    :return: Binned data as integer indices.
    """
    assert len(data.shape) > 1, "Input data must have at least two dimensions."
    assert data.shape[-1] > 0, "Input data must have at least one feature."

    def _bin_feature_uniformly(col: jnp.ndarray) -> jnp.ndarray:
        """
        Bin a single feature (all entries for one position on the last axis) uniformly.
        `col` has shape (...,), and we treat all its elements together.
        """
        # Compute bin edges for this feature
        min_value = col.min() - 1e-3  # Widen edges a bit to prevent numerical issues
        max_value = col.max() + 1e-3
        bin_edges = jnp.linspace(min_value, max_value, num_bins + 1)

        # Digitize this feature's data into bins
        binned = jnp.digitize(col, bin_edges) - 1  # 0-based indices

        # Clip values to ensure they fall within valid range
        binned = jnp.clip(binned, 0, num_bins - 1)
        return binned

    # Move last axis to front: (..., D) -> (D, ...)
    data_swapped = jnp.moveaxis(data, -1, 0)

    # vmap over the feature axis (each feature gets its own binning)
    binned_swapped = jax.vmap(_bin_feature_uniformly, in_axes=0, out_axes=0)(data_swapped)

    # Move axes back: (D, ...) -> (..., D)
    binned_data = jnp.moveaxis(binned_swapped, 0, -1)
    return binned_data


def bin_into_equal_population_bins(data: jnp.ndarray, num_bins: int) -> jnp.ndarray:
    """
    Bin the input data into bins with approximately equal population. Since we are using ReLUs, the data
    may include a lot of zeros. We put those into a separate bin, if there are any such points. The remaining
    data points are then binned into the remaining bins based on quantiles.
    DISCLAIMER: This code was written by ChatGPT. I tried to test it thoroughly, but I am not sure if it is bug-free.
    :param data: Input data to be binned.
    :param num_bins: Number of bins to create.
    :return: Binned data as integer indices.
    """
    assert len(data.shape) > 1, "Input data must have at least two dimensions."
    assert data.shape[-1] > 0, "Input data must have at least one feature."

    def bin_equal_population_with_ties_jax_1d(x: jnp.ndarray) -> jnp.ndarray:
        """
        Tie-respecting equal-population binning for 1D data in JAX.

        Returns binned indices of same shape as x, with bins 0..(k-1),
        where k <= num_bins if there aren't enough distinct cut positions.
        """
        x = jnp.asarray(x).ravel()
        N = x.shape[0]

        # Degenerate case: everything in bin 0
        if num_bins <= 1 or N == 0:
            return jnp.zeros_like(x, dtype=jnp.int32)

        # 1) Sort the data
        x_sorted = jnp.sort(x)

        # 2) Candidate cut positions: indices i where x_sorted[i] != x_sorted[i+1]
        #    We represent candidates with a boolean mask over indices 0..N-1.
        #    Last index can never be a cut position.
        if N > 1:
            diffs = x_sorted[:-1] != x_sorted[1:]  # shape (N-1,)
            candidate_mask = jnp.concatenate(
                [diffs, jnp.array([False], dtype=bool)]
            )  # shape (N,)
        else:
            candidate_mask = jnp.array([False], dtype=bool)

        idxs_all = jnp.arange(N, dtype=jnp.int32)
        cum_probs = (idxs_all + 1) / N  # (i+1)/N

        # 3) Target cumulative probabilities for bins
        n_cuts = num_bins - 1
        targets = jnp.linspace(1.0 / num_bins,
                               1.0 - 1.0 / num_bins,
                               n_cuts)

        # 4) Sequential "nearest unused" selection of cut indices
        init_prev_idx = jnp.int32(-1)
        init_cut_indices = jnp.full((n_cuts,), -1, dtype=jnp.int32)

        def body_fun(i, carry):
            prev_idx, cut_indices = carry
            t = targets[i]

            start = prev_idx + 1

            # Valid candidates: not yet used (>= start), not last index, and a cut candidate
            valid = (idxs_all >= start) & (idxs_all < (N - 1)) & candidate_mask

            # Distance in probability space; invalid positions get +inf
            diffs = jnp.where(valid, jnp.abs(cum_probs - t), jnp.inf)

            best_idx = jnp.argmin(diffs)
            any_valid = jnp.any(valid)

            best_idx = jnp.where(any_valid, best_idx, prev_idx)
            cut_indices = cut_indices.at[i].set(jnp.where(any_valid, best_idx, -1))
            prev_idx = jnp.where(any_valid, best_idx, prev_idx)

            return (prev_idx, cut_indices)

        _, cut_indices = lax.fori_loop(0, n_cuts, body_fun,
                                       (init_prev_idx, init_cut_indices))

        # 5) Convert cut indices to edge values
        #    -1 means "no cut here"; we map those to +inf so they never affect binning.
        clipped = jnp.clip(cut_indices, 0, jnp.maximum(N - 1, 0))
        inner_edges_all = x_sorted[clipped]
        valid_mask = cut_indices >= 0
        inner_edges = jnp.where(valid_mask, inner_edges_all, jnp.inf)

        # 6) Assign bins:
        #    bins = number of edges strictly less than x
        #         = sum(x > edge_k over k)
        cmp = x[:, None] > inner_edges[None, :]  # shape (N, n_cuts)
        bins = jnp.sum(cmp, axis=1).astype(jnp.int32)

        return bins.reshape(x.shape)

    # Move last axis to front: (..., D) -> (D, ...)
    data_swapped = jnp.moveaxis(data, -1, 0)

    # vmap over the feature axis (each feature gets its own binning)
    binned_swapped = jax.vmap(bin_equal_population_with_ties_jax_1d, in_axes=0, out_axes=0)(data_swapped)

    # Move axes back: (D, ...) -> (..., D)
    binned_data = jnp.moveaxis(binned_swapped, 0, -1)
    return binned_data
