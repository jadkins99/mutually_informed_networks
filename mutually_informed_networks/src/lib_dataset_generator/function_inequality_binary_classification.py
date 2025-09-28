import math
from typing import Callable

import jax.numpy as jnp
import jax.random as jrandom


def get_linear_dataset(dataset_size: int, key: jrandom.PRNGKey, noise_std_error: float = 0.5) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate a dataset for binary classification, where the inputs are the inputs x, and x with some noise added.
    The targets are 1 if the noisy x is less than the true x, and 0 otherwise.
    :param dataset_size: Integer, the number of samples to generate.
    :param key: Random key / seed for JAX.
    :param noise_std_error: Magnitude of the noise to add to x.
    :return: a pair od arrays of inputs (x and noisy x) and targets (0 or 1).
    """
    return get_dataset_for_function(lambda x: x, dataset_size, key, noise_std_error)


def get_sin_dataset(dataset_size: int, key: jrandom.PRNGKey, noise_std_error: float = 0.5) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate a dataset for binary classification, where the inputs are the inputs x, and sin(x) with some noise added.
    The targets are 1 if the noisy sin(x) is less than the true sin(x)), and 0 otherwise.
    :param dataset_size: Integer, the number of samples to generate.
    :param key: Random key / seed for JAX.
    :param noise_std_error: Magnitude of the noise to add to sin(x).
    :return: a pair od arrays of inputs (x and noisy sin(x)) and targets (0 or 1).
    """
    return get_dataset_for_function(jnp.sin, dataset_size, key, noise_std_error)


def get_dataset_for_function(function: Callable, dataset_size: int, key: jrandom.PRNGKey, noise_std_error: float = 0.5) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate a dataset for binary classification, where the inputs are the inputs x, and f(x) with some noise added.
    The targets are 1 if the noisy f(x) is less than the true f(x), and 0 otherwise.
    :param function: Callable, the function to apply to x (e.g., jnp.sin).
    :param dataset_size: Integer, the number of samples to generate.
    :param key: Random key / seed for JAX.
    :param noise_std_error: Magnitude of the noise to add to f(x).
    :return: a pair od arrays of inputs (x and noisy f(x)) and targets (0 or 1).
    """
    x = jnp.expand_dims(jnp.linspace(0, 20 * math.pi, dataset_size), axis=1)
    func_value = function(x)

    func_value_with_noise = func_value + jrandom.normal(key, (dataset_size, 1)) * noise_std_error

    input_data = jnp.concatenate([x, func_value_with_noise], axis=1)
    target_data = func_value_with_noise < func_value

    return input_data, target_data
