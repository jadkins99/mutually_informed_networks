import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np

from mi_plane import plot_mi_plane
from mlp import MLP
from mod_n_classification import get_mod_n_dataset
from trajectory import train_on_dataset, TrajectoryInfo


def get_trajectory_losses(seed: int, num_classes: int, num_permutations: int) -> list[TrajectoryInfo]:
    data_key, model_key = jrandom.split(jrandom.PRNGKey(seed=0), 2)
    xs, ys = get_mod_n_dataset(dataset_size=256, key=data_key, n=num_classes)
    model = MLP(in_size=xs[0].shape[-1], out_size=ys[0].shape[-1], layer_sizes=[4, 3, 2], key=model_key)

    trajectory_infos: list[TrajectoryInfo] = []
    for traj_idx in range(num_permutations):
        print(f"Training trajectory {traj_idx} with seed {traj_idx + seed}...")
        data_key = jrandom.split(jrandom.PRNGKey(seed=traj_idx + seed), 1)[0]
        xs, ys = get_mod_n_dataset(dataset_size=256, key=data_key, n=num_classes, shuffle_bits=True)

        model, trajectory_info = train_on_dataset(
            dataset=(xs, ys),
            model=model,
            batch_size=32,
            learning_rate=1e-3,
            steps=1000,
            compute_mi_every=100,
        )
        trajectory_infos.append(trajectory_info)

    return trajectory_infos


if __name__ == "__main__":
    num_seeds = 2
    num_classes = 4
    num_permutations = 10

    traj_infos_per_seed: list[list[TrajectoryInfo]] = []
    for seed in range(2):
        trajectory_infos = get_trajectory_losses(seed, num_classes, num_permutations)
        traj_infos_per_seed.append(trajectory_infos)

    final_loss_sp = np.array([[t.losses[-1] for t in t_infos] for t_infos in traj_infos_per_seed])  # shape (num_seeds, num_permutations)
    mean = np.array(final_loss_sp).mean(0)
    ste = np.array(final_loss_sp).std(0) / np.sqrt(len(traj_infos_per_seed))
    plt.grid(alpha=0.2)
    plt.axhline(y=np.log(4), color='grey', linestyle='--', label='Random Guess Loss')
    plt.plot(mean)
    plt.fill_between(range(len(mean)), mean - ste, mean + ste, alpha=0.3)
    plt.ylim(ymin=0)
    plt.xlabel('Trajectory Index')
    plt.ylabel('Final Loss')
    plt.title('Final Losses Across Different Input Bit Permutations')
    plt.savefig('final_losses.png')

    final_accuracy_sp = np.array([[t.accuracies[-1] for t in t_infos] for t_infos in traj_infos_per_seed])  # shape (num_seeds, num_permutations)
    mean = np.array(final_accuracy_sp).mean(0)
    ste = np.array(final_accuracy_sp).std(0) / np.sqrt(len(traj_infos_per_seed))
    plt.clf()
    plt.grid(alpha=0.2)
    plt.axhline(y=1 / num_classes, color='grey', linestyle='--', label='Random Guess Accuracy')
    plt.plot(mean)
    plt.fill_between(range(len(mean)), mean - ste, mean + ste, alpha=0.3)
    plt.ylim(0, 1.1)
    plt.xlabel('Trajectory Index')
    plt.ylabel('Final Accuracy')
    plt.title('Final Accuracies Across Different Input Bit Permutations')
    plt.savefig('final_accuracies.png')

    mi_inputs_spln = np.array([[t.mi_with_input for t in t_infos] for t_infos in traj_infos_per_seed])  # shape (num_seeds, num_permutations, num_layers, num_steps)
    mi_outputs_spln = np.array([[t.mi_with_output for t in t_infos] for t_infos in traj_infos_per_seed])  # shape (num_seeds, num_permutations, num_layers, num_steps)
    mi_inputs_pln = np.array(mi_inputs_spln).mean(0)
    mi_outputs_pln = np.array(mi_outputs_spln).mean(0)
    for traj_idx in range(len(mi_inputs_pln)):
        mi_input = mi_inputs_pln[traj_idx]
        mi_output = mi_outputs_pln[traj_idx]
        plot_mi_plane(mi_input, mi_output, idx=traj_idx)
