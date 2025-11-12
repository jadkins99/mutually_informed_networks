import jax.numpy as jnp


def compute_accuracy(pred_ys: jnp.ndarray, ys: jnp.ndarray) -> float:
    """Compute the accuracy of predictions against true labels.

    Args:
        pred_ys (jnp.ndarray): Predicted outputs from the model.
        ys (jnp.ndarray): True labels.

    Returns:
        float: The accuracy as a float between 0 and 1.
    """
    num_correct = jnp.sum(jnp.argmax(pred_ys, axis=-1) == jnp.argmax(ys, axis=-1))
    accuracy = (num_correct / ys.shape[0]).item()
    return accuracy