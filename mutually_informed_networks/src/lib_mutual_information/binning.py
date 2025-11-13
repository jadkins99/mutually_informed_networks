import jax
import jax.numpy as jnp



def bin_into_uniform_bins(data: jnp.ndarray, num_bins: int):
    """
    Bin the input data into uniform bins.
    :param data: Input data to be binned.
    :param num_bins: Number of bins to create.
    :return: Binned data as integer indices.
    """
    assert len(data.shape) > 1, "Input data must have at least two dimensions."

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
    :param data: Input data to be binned.
    :param num_bins: Number of bins to create.
    :return: Binned data as integer indices.
    """
    assert len(data.shape) > 1, "Input data must have at least two dimensions."

    def bin_one_feature(col: jnp.ndarray) -> jnp.ndarray:
        zero_mask = (col == 0)
        has_zero = jnp.any(zero_mask)

        def bin_without_zeros(col_: jnp.ndarray) -> jnp.ndarray:
            # Original behaviour: bin all values into 0..num_bins-1 via quantiles
            qs = jnp.linspace(0.0, 1.0, num_bins + 1)
            edges = jnp.quantile(col_, qs)
            edges = edges.at[0].add(-1e-3)
            edges = edges.at[-1].add(1e-3)
            binned = jnp.digitize(col_, edges) - 1
            return binned.astype(jnp.int32)

        def bin_with_zeros(col_: jnp.ndarray) -> jnp.ndarray:
            # If everything is zero, just return zeros
            has_nonzero = jnp.any(col_ != 0)

            def some_nonzeros(col2: jnp.ndarray) -> jnp.ndarray:
                remaining_bins = num_bins - 1

                # Replace zeros by NaN so they are ignored in nanquantile
                col_for_q = jnp.where(col2 == 0, jnp.nan, col2)

                qs = jnp.linspace(0.0, 1.0, remaining_bins + 1)
                edges_nz = jnp.nanquantile(col_for_q, qs)

                # Widen edges a bit
                edges_nz = edges_nz.at[0].add(-1e-3)
                edges_nz = edges_nz.at[-1].add(1e-3)

                # Digitize *all* values with non-zero edges
                # This gives bins 0..remaining_bins-1 for non-zeros
                # (zeros may get something weird; we'll override them)
                binned_nz = jnp.digitize(col2, edges_nz) - 1

                # Shift to global bins 1..num_bins-1
                binned_global = binned_nz + 1

                # Force zeros to bin 0
                binned_global = jnp.where(col2 == 0, 0, binned_global)
                return binned_global.astype(jnp.int32)

            def no_nonzeros(col2: jnp.ndarray) -> jnp.ndarray:
                # All values are zero
                return jnp.zeros_like(col2, dtype=jnp.int32)

            return jax.lax.cond(has_nonzero, some_nonzeros, no_nonzeros, col_)

        # Choose branch based on presence of zeros
        return jax.lax.cond(has_zero, bin_with_zeros, bin_without_zeros, col)

    # Move last axis to front: (..., D) -> (D, ...)
    data_swapped = jnp.moveaxis(data, -1, 0)

    # vmap over the feature axis (each feature gets its own binning)
    binned_swapped = jax.vmap(bin_one_feature, in_axes=0, out_axes=0)(data_swapped)

    # Move axes back: (D, ...) -> (..., D)
    binned_data = jnp.moveaxis(binned_swapped, 0, -1)
    return binned_data
