import jax
import jax.numpy as jnp
import equinox as eqx


@eqx.filter_value_and_grad
def compute_bce_loss(model, x, y):
    pred_y = jax.vmap(model)(x)
    # Trains with respect to binary cross-entropy
    return -jnp.mean(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y))