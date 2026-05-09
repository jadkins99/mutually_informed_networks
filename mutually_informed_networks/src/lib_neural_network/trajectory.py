from dataclasses import dataclass, Field
from typing import Callable

import equinox as eqx
import jax
import optax
from jax import numpy as jnp

from cross_entropy import compute_ce_loss
from dataloader import dataloader
from drop_neuron_mi import mutual_information_without_neuron
from layer_wise_mi import compute_layer_wise_mi_with_quantile_binning
from metrics import compute_accuracy
from mlp import MLP


class TrajectoryInfo:
    def __init__(self, model: MLP):
        self._num_layers = len(model.layers)
        self.losses: list[float] = []
        self.accuracies: list[float] = []

        self.mi_input: dict[str, list[float]] = {f"Layer {i}": [] for i in range(self._num_layers)}
        self.mi_output: dict[str, list[float]] = {f"Layer {i}": [] for i in range(self._num_layers)}
        self.mi_input_wo: dict[str, list[float]] = {f"Layer {i}, neuron {j}": []
                                                    for i in range(self._num_layers)
                                                    for j in range(model.layers[i].bias.shape[0])}
        self.mi_output_wo: dict[str, list[float]] = {f"Layer {i}, neuron {j}": []
                                                     for i in range(self._num_layers)
                                                     for j in range(model.layers[i].bias.shape[0])}

    def update_mi(self, mi_with_input: list[float], mi_with_output: list[float]):
        assert len(mi_with_input) == self._num_layers, f"Mismatch: {len(mi_with_input)} != {self._num_layers}"
        for layer_idx in range(len(mi_with_input)):
            self.mi_input[f"Layer {layer_idx}"].append(mi_with_input[layer_idx])
            self.mi_output[f"Layer {layer_idx}"].append(mi_with_output[layer_idx])

    def update_mi_wo(self, layer_idx: int, neuron_idx: int, mi_input_wo: float, mi_output_wo: float):
        self.mi_input_wo[f"Layer {layer_idx}, neuron {neuron_idx}"].append(mi_input_wo)
        self.mi_output_wo[f"Layer {layer_idx}, neuron {neuron_idx}"].append(mi_output_wo)

    @property
    def mi_with_input(self):
        return [self.mi_input[f"Layer {layer_idx}"] for layer_idx in range(self._num_layers)]

    @property
    def mi_with_output(self):
        return [self.mi_output[f"Layer {layer_idx}"] for layer_idx in range(self._num_layers)]


def _update_trajectory_info(
        trajectory_info: TrajectoryInfo, model: eqx.Module, dataset: tuple[jnp.ndarray, jnp.ndarray], num_bins: int = 3):
    mi_with_input, mi_with_output = compute_layer_wise_mi_with_quantile_binning(
        model, dataset[0], dataset[1], num_bins=num_bins)
    trajectory_info.update_mi(mi_with_input, mi_with_output)

    for layer_idx in range(len(model.layers)):
        for neuron_idx in range(model.layers[layer_idx].weight.shape[0]):
            mi_with_input_n, mi_with_output_n = mutual_information_without_neuron(
                model, layer_idx, neuron_idx, dataset[0], dataset[1], num_bins=num_bins)
            trajectory_info.update_mi_wo(layer_idx, neuron_idx, mi_with_input_n[layer_idx], mi_with_output_n[layer_idx])


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
    model: MLP,
    batch_size: int = 32,
    learning_rate: float = 3e-3,
    steps: int = 200,
    compute_mi_every: int = 25,
    callbacks: list[Callable] = None,
):
    iter_data = dataloader(dataset, batch_size)

    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)
    trajectory_info = TrajectoryInfo(model)

    for step, (x, y) in zip(range(steps), iter_data):
        loss, model, opt_state = make_step(model, x, y, optim, opt_state)
        trajectory_info.losses.append(loss.item())
        trajectory_info.accuracies.append(compute_accuracy(ys=dataset[1], pred_ys=jax.vmap(model)(dataset[0])))

        if compute_mi_every > 0 and (step % compute_mi_every == 0 or (step + 1) == steps):
            _update_trajectory_info(trajectory_info, model, dataset)

            for callback in callbacks or []:
                model = callback(model, dataset[0], dataset[1])

    return model, trajectory_info
