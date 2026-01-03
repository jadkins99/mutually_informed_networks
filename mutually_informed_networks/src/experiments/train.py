import jax
import jax.numpy as jnp
import equinox as eqx
import jax.random as jrandom
import optax

from dataloader import dataloader
from drop_neuron_mi import mutual_information_without_neuron, DropNeuronResetHeuristicCallback
from function_inequality_binary_classification import get_linear_dataset
from layer_wise_mi import compute_layer_wise_mi_with_uniform_binning, compute_layer_wise_mi_with_quantile_binning
from metrics import compute_accuracy
from mi_plane import plot_mi_plane
from mlp import MLP
from cross_entropy import compute_bce_loss, compute_ce_loss
from mod_n_classification import get_mod_n_dataset


# Important for efficiency whenever you use JAX: wrap everything into a single JIT region.
@eqx.filter_jit
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
    compute_mi_every: int = 25,
):
    iter_data = dataloader(dataset, batch_size)

    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)
    losses = []
    mi_with_input_history, mi_with_output_history = [], []
    mi_input_diffs, mi_output_diffs = {}, {}

    model_reset_callback = DropNeuronResetHeuristicCallback(num_bins=3, reset_threshold=0.1)
    for step, (x, y) in zip(range(steps), iter_data):
        loss, model, opt_state = make_step(model, x, y, optim, opt_state)
        loss = loss.item()
        losses.append(loss)

        if compute_mi_every > 0 and (step % compute_mi_every == 0 or (step + 1) == steps):
            mi_with_input, mi_with_output = compute_layer_wise_mi_with_quantile_binning(
                model, dataset[0], dataset[1], num_bins=3)
            mi_with_input_history.append(mi_with_input)
            mi_with_output_history.append(mi_with_output)

            mi_input_diffs = {f"Layer {i}": [] for i in range(len(mi_with_input))}
            mi_output_diffs = {f"Layer {i}": [] for i in range(len(mi_with_input))}
            for layer_idx in range(len(model.layers)):
                for neuron_idx in range(model.layers[layer_idx].weight.shape[0]):
                    mi_with_input_n, mi_with_output_n = mutual_information_without_neuron(
                        model, layer_idx, neuron_idx, dataset[0], dataset[1], num_bins=3)
                    mi_input_diffs[f"Layer {layer_idx}"].append(mi_with_input[layer_idx] - mi_with_input_n[layer_idx])
                    mi_output_diffs[f"Layer {layer_idx}"].append(mi_with_output[layer_idx] - mi_with_output_n[layer_idx])

            model = model_reset_callback(model, dataset[0], dataset[1])

    final_accuracy = compute_accuracy(ys=dataset[1], pred_ys=jax.vmap(model)(dataset[0]))
    print(model_reset_callback.mean_mi_drop_per_neuron)
    return model, losses, final_accuracy, mi_with_input_history, mi_with_output_history, mi_input_diffs, mi_output_diffs


if __name__ == "__main__":
    mi_inputs, mi_outputs = [], []
    for seed in range(100):
        data_key, model_key = jrandom.split(jrandom.PRNGKey(seed=seed), 2)
        xs, ys = get_mod_n_dataset(dataset_size=1_000, key=data_key, n=4)
        model = MLP(in_size=xs[0].shape[-1], out_size=ys[0].shape[-1], layer_sizes=[12, 10, 8, 6], key=model_key)

        model, losses, final_accuracy, mi_input, mi_output = train_on_dataset(
            dataset=(xs, ys),
            model=model,
            batch_size=32,
            learning_rate=1e-3,
            steps=400,
        )
        mi_inputs.append(mi_input)
        mi_outputs.append(mi_output)

    mi_input = jnp.mean(jnp.array(mi_inputs), axis=0).tolist()
    mi_output = jnp.mean(jnp.array(mi_outputs), axis=0).tolist()
    plot_mi_plane(mi_input, mi_output)