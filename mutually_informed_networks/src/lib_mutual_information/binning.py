import numpy as np


def bin_into_uniform_bins(data: np.ndarray, num_bins: int) -> np.ndarray:
    """
    NumPy version of bin_into_uniform_bins.

    Uniform binning along the last axis. Each feature gets its own bin edges
    computed from its min/max values.

    :param data: Input array of shape (..., D)
    :param num_bins: Number of uniform-width bins
    :return: Binned integer array of same shape as `data`, dtype int32
    """
    data = np.asarray(data)
    assert len(data.shape) > 1, "Input data must have at least two dimensions."
    assert data.shape[-1] > 0, "Input data must have at least one feature."

    def _bin_feature_uniformly(col: np.ndarray) -> np.ndarray:
        """
        Bin a single feature (vector of shape (N,)).
        """
        col = np.asarray(col)
        min_value = col.min() - 1e-3
        max_value = col.max() + 1e-3

        # Uniform bin edges
        bin_edges = np.linspace(min_value, max_value, num_bins + 1)

        # Digitize produces values in [1, num_bins+1]
        binned = np.digitize(col, bin_edges) - 1

        # Clip to ensure valid 0..num_bins-1 range
        return np.clip(binned, 0, num_bins - 1).astype(np.int32)

    # Move last axis to front: (..., D) -> (D, ...)
    data_swapped = np.moveaxis(data, -1, 0)

    # Allocate output
    binned_swapped = np.empty_like(data_swapped, dtype=np.int32)

    # Bin each feature independently
    for i in range(data_swapped.shape[0]):
        binned_swapped[i] = _bin_feature_uniformly(data_swapped[i])

    # Move axes back: (D, ...) -> (..., D)
    return np.moveaxis(binned_swapped, 0, -1)


def bin_into_equal_population_bins(data: np.ndarray, num_bins: int) -> np.ndarray:
    """
    NumPy version of bin_into_equal_population_bins.

    Bins the last axis of `data` into ~equal-population bins separately for each feature,
    using tie-respecting binning along the batch axis.

    :param data: Input data, shape (..., D).
    :param num_bins: Number of bins to create.
    :return: Binned data as integer indices, same shape as `data`.
    """

    def bin_equal_population_with_ties_1d(x: np.ndarray, num_bins: int) -> np.ndarray:
        """
        Tie-respecting equal-population binning for 1D data in NumPy.

        Returns binned indices of same shape as x, with bins 0..(k-1),
        where k <= num_bins if there aren't enough distinct cut positions.
        """
        x = np.asarray(x)
        original_shape = x.shape
        x = x.ravel()
        N = x.shape[0]

        # Degenerate case: everything in bin 0
        if num_bins <= 1 or N == 0:
            return np.zeros_like(x, dtype=np.int32).reshape(original_shape)

        # 1) Sort the data
        x_sorted = np.sort(x)

        # 2) Candidate cut positions: indices i where x_sorted[i] != x_sorted[i+1]
        #    Last index can never be a cut position.
        if N > 1:
            diffs = x_sorted[:-1] != x_sorted[1:]  # shape (N-1,)
            candidate_mask = np.concatenate(
                [diffs, np.array([False], dtype=bool)]
            )  # shape (N,)
        else:
            candidate_mask = np.array([False], dtype=bool)

        idxs_all = np.arange(N, dtype=np.int32)
        cum_probs = (idxs_all + 1) / N  # (i+1)/N

        # 3) Target cumulative probabilities for bins
        n_cuts = num_bins - 1
        targets = np.linspace(1.0 / num_bins,
                              1.0 - 1.0 / num_bins,
                              n_cuts)

        # 4) Sequential "nearest unused" selection of cut indices
        prev_idx = np.int32(-1)
        cut_indices = np.full((n_cuts,), -1, dtype=np.int32)

        for i in range(n_cuts):
            t = targets[i]
            start = prev_idx + 1

            # Valid candidates: not yet used (>= start), not last index, and a cut candidate
            valid = (idxs_all >= start) & (idxs_all < (N - 1)) & candidate_mask

            if not np.any(valid):
                # No more valid cuts; leave cut_indices[i:] as -1
                break

            # Distance in probability space; invalid positions get +inf
            diffs_prob = np.where(valid, np.abs(cum_probs - t), np.inf)

            best_idx = np.argmin(diffs_prob)
            cut_indices[i] = best_idx
            prev_idx = best_idx

        # 5) Convert cut indices to edge values
        #    -1 means "no cut here"; we map those to +inf so they never affect binning.
        if N > 0:
            clipped = np.clip(cut_indices, 0, N - 1)
        else:
            clipped = cut_indices
        inner_edges_all = x_sorted[clipped]
        valid_mask = cut_indices >= 0
        inner_edges = np.where(valid_mask, inner_edges_all, np.inf)

        # 6) Assign bins:
        #    bins = number of edges strictly less than x
        #         = sum(x > edge_k over k)
        cmp = x[:, None] > inner_edges[None, :]  # shape (N, n_cuts)
        bins = np.sum(cmp, axis=1).astype(np.int32)

        return bins.reshape(original_shape)

    data = np.asarray(data)
    assert len(data.shape) > 1, "Input data must have at least two dimensions."
    assert data.shape[-1] > 0, "Input data must have at least one feature."

    # Move last axis to front: (..., D) -> (D, ...)
    data_swapped = np.moveaxis(data, -1, 0)

    # Allocate output with int32 dtype
    binned_swapped = np.empty_like(data_swapped, dtype=np.int32)

    # For each feature, bin all samples jointly
    for i in range(data_swapped.shape[0]):
        binned_swapped[i] = bin_equal_population_with_ties_1d(
            data_swapped[i], num_bins=num_bins
        )

    # Move axes back: (D, ...) -> (..., D)
    binned_data = np.moveaxis(binned_swapped, 0, -1)
    return binned_data
