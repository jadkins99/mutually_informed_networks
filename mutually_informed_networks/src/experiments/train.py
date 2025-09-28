import jax
import jax.numpy as jnp
import equinox as eqx
import jax.random as jrandom
import optax

from dataloader import dataloader
from function_inequality_binary_classification import get_linear_dataset
from mlp import MLP
from binary_cross_entropy import compute_bce_loss


# Important for efficiency whenever you use JAX: wrap everything into a single JIT region.
# @eqx.filter_jit
def make_step(
        model: eqx.Module,
        x: jnp.ndarray,
        y: jnp.ndarray,
        optim: optax.GradientTransformation,
        opt_state: optax.OptState
):
    loss, grads = compute_bce_loss(model, x, y)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def train_on_dataset(
    dataset: tuple[jnp.ndarray, jnp.ndarray],
    model: eqx.Module,
    batch_size: int = 32,
    learning_rate: float = 3e-3,
    steps: int = 200,
):
    iter_data = dataloader((xs, ys), batch_size)
    dataset_size = dataset[0].shape[0]

    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)
    losses = []
    for step, (x, y) in zip(range(steps), iter_data):
        loss, model, opt_state = make_step(model, x, y, optim, opt_state)
        loss = loss.item()
        print(f"step={step}, loss={loss}")
        losses.append(loss)

    pred_ys = jax.vmap(model)(xs)
    num_correct = jnp.sum((pred_ys > 0.5) == ys)
    final_accuracy = (num_correct / dataset_size).item()
    print(f"final_accuracy={final_accuracy}")
    return model, xs, ys, pred_ys, losses


if __name__ == "__main__":
    data_key, model_key = jrandom.split(jrandom.PRNGKey(seed=42), 2)
    xs, ys = get_linear_dataset(dataset_size=10_000, key=data_key)
    model = MLP(in_size=2, out_size=1, width_size=32, depth=3, key=model_key)

    model, xs, ys, pred_ys, losses = train_on_dataset(
        dataset=(xs, ys),
        model=model,
        batch_size=32,
        learning_rate=3e-3,
        steps=200,
    )