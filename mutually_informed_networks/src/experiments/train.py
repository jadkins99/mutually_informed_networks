import jax
import jax.numpy as jnp
import equinox as eqx
import jax.random as jrandom
import optax

from dataloader import dataloader
from function_inequality_binary_classification import get_linear_dataset
from layer_wise_mi import compute_layer_wise_mi_with_uniform_binning
from metrics import compute_accuracy
from mi_plane import plot_mi_plane
from mlp import MLP
from cross_entropy import compute_bce_loss, compute_ce_loss
from mod_n_classification import get_mod_n_dataset


# Important for efficiency whenever you use JAX: wrap everything into a single JIT region.
# @eqx.filter_jit
def make_step(
        model: eqx.Module,
        x: jnp.ndarray,
        y: jnp.ndarray,
        optim: optax.GradientTransformation,
        opt_state: optax.OptState
):
    loss, grads = compute_ce_loss(model, x, y)
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
    iter_data = dataloader(dataset, batch_size)

    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)
    losses = []
    mi_with_input_history, mi_with_output_history = [], []
    for step, (x, y) in zip(range(steps), iter_data):
        loss, model, opt_state = make_step(model, x, y, optim, opt_state)
        loss = loss.item()
        print(f"step={step}, loss={loss}")
        losses.append(loss)
        if (step + 1) % 10 == 0 and step > 50:
            mi_with_input, mi_with_output = compute_layer_wise_mi_with_uniform_binning(
                model, dataset[0], dataset[1], num_bins=30)
            mi_with_input_history.append(mi_with_input)
            mi_with_output_history.append(mi_with_output)

    pred_ys = jax.vmap(model)(xs)
    final_accuracy = compute_accuracy(ys=ys, pred_ys=pred_ys)
    print(f"final_accuracy={final_accuracy}")
    return model, xs, ys, pred_ys, losses, mi_with_input_history, mi_with_output_history


if __name__ == "__main__":
    data_key, model_key = jrandom.split(jrandom.PRNGKey(seed=42), 2)
    xs, ys = get_mod_n_dataset(dataset_size=10_000, key=data_key, n=4)
    model = MLP(in_size=xs[0].shape[-1], out_size=ys[0].shape[-1], width_size=8, depth=3, key=model_key)

    model, xs, ys, pred_ys, losses, mi_input, mi_output = train_on_dataset(
        dataset=(xs, ys),
        model=model,
        batch_size=32,
        learning_rate=3e-3,
        steps=200,
    )
    plot_mi_plane(mi_input, mi_output)