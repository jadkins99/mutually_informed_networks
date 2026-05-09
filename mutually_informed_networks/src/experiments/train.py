import jax.numpy as jnp
import jax.random as jrandom

from drop_neuron_mi import DropNeuronResetHeuristicCallback
from mi_plane import plot_mi_plane
from mlp import MLP
from mod_n_classification import get_mod_n_dataset
from trajectory import train_on_dataset, TrajectoryInfo

if __name__ == "__main__":
    trajectory_infos: list[TrajectoryInfo] = []
    for seed in range(1):
        print(f'Training trajectory with seed {seed}...')
        data_key, model_key = jrandom.split(jrandom.PRNGKey(seed=seed), 2)
        xs, ys = get_mod_n_dataset(dataset_size=1_000, key=data_key, n=4)
        model = MLP(in_size=xs[0].shape[-1], out_size=ys[0].shape[-1], layer_sizes=[10, 8, 6], key=model_key)

        model, trajectory_info = train_on_dataset(
            dataset=(xs, ys),
            model=model,
            batch_size=32,
            learning_rate=1e-3,
            steps=400,
            callbacks=[DropNeuronResetHeuristicCallback(num_bins=3, reset_threshold=0.0)]
        )
        trajectory_infos.append(trajectory_info)

    mi_input = jnp.array([t.mi_with_input for t in trajectory_infos]).mean(axis=0).tolist()
    mi_output = jnp.array([t.mi_with_output for t in trajectory_infos]).mean(axis=0).tolist()
    plot_mi_plane(mi_input, mi_output)