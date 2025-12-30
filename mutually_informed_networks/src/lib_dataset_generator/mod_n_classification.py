import jax.numpy as jnp
import jax.random as jrandom


def get_single_bit_dataset(match: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
    """ Returns a mapping from [0, 1] -> [0, 1] or [0, 1] -> [1, 0] depending on the value of `match`. """
    x = jnp.array([[0], [1]], dtype=jnp.float32)
    if match:
        y = jnp.array([[1, 0], [0, 1]], dtype=jnp.int32)
    else:
        y = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
    return x, y


def get_mod_n_dataset(
        dataset_size: int, key: jrandom.PRNGKey, n: int = 2, shuffle_bits: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate a dataset for n-class classification, where the inputs are the inputs x, and the targets are x mod n.
    The number x is a 8-bit integer, so a number in [0, 255].
    :param dataset_size: Number of samples to be generated.
    :param key: Random key / seed for JAX.
    :param n: Number of classes, or mod of the inputs.
    :param shuffle_bits: Whether to randomly permute the input bits and possibly flip the target classes.
    :return: a pair of arrays of inputs and targets (0, 1, ..., n-1).
    """
    assert n < 2**8, f'Requested number of classes {n} is too large for 8-bit representation.'
    if dataset_size > 2**8:
        print(f"Warning: dataset_size {dataset_size} is larger than the number of unique 8-bit integers (256). "
              f"Enumerating the whole space instead of sampling randomly.")
        x = jnp.arange(0, 2**8).reshape(-1, 1)[jrandom.permutation(key, jnp.arange(2**8))]
    else:
        x = jrandom.randint(key, (dataset_size, 1), minval=0, maxval=2**8)
    y = x % n
    x_bits = jnp.unpackbits(x.astype('uint8')).reshape(len(x), 8)
    y_one_hot = jnp.eye(n)[y.flatten()]

    if shuffle_bits:
        x_perm_key, y_perm_key = jrandom.split(key, num=2)
        x_perm = jrandom.permutation(x_perm_key, jnp.arange(8))
        x_bits = x_bits[:, x_perm]

        y_perm = jrandom.permutation(y_perm_key, jnp.arange(n))
        y_one_hot = y_one_hot[:, y_perm]

    return x_bits.astype(jnp.float32), y_one_hot.astype(jnp.int32)