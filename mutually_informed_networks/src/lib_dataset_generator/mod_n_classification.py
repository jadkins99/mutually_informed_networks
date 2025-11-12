import jax.numpy as jnp
import jax.random as jrandom


def get_mod_n_dataset(dataset_size: int, key: jrandom.PRNGKey, n: int = 2) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate a dataset for n-class classification, where the inputs are the inputs x, and the targets are x mod n.
    The number x is a 8-bit integer, so a number in [0, 255].
    :param dataset_size: Number of samples to be generated.
    :param key: Random key / seed for JAX.
    :param n: Number of classes, or mod of the inputs.
    :return: a pair of arrays of inputs and targets (0, 1, ..., n-1).
    """
    assert n < 2**8, f'Requested number of classes {n} is too large for 8-bit representation.'
    x = jrandom.randint(key, (dataset_size, 1), minval=0, maxval=2**16)
    y = x % n
    x_bits = jnp.unpackbits(x.astype('uint8')).reshape(len(x), 8)
    y_one_hot = jnp.eye(n)[y.flatten()]
    return x_bits.astype(jnp.float32), y_one_hot.astype(jnp.int32)