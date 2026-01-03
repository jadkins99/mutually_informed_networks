import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np

from mi_plane import plot_mi_plane
from mlp import MLP
from mod_n_classification import get_mod_n_dataset
from mutually_informed_networks.src.experiments.train import train_on_dataset


def get_trajectory_losses(seed):
    data_key, model_key = jrandom.split(jrandom.PRNGKey(seed=0), 2)
    xs, ys = get_mod_n_dataset(dataset_size=256, key=data_key, n=4)
    model = MLP(in_size=xs[0].shape[-1], out_size=ys[0].shape[-1], layer_sizes=[4, 3, 2], key=model_key)

    final_losses = []
    final_accuracies = []
    mi_inputs, mi_outputs = [], []
    mi_input_diffss, mi_output_diffss = [], []
    for step in range(20):
        print(f"Training trajectory {step} with seed {step + seed}...")
        data_key = jrandom.split(jrandom.PRNGKey(seed=step + seed), 1)[0]
        xs, ys = get_mod_n_dataset(dataset_size=256, key=data_key, n=4, shuffle_bits=True)

        compute_mi = True#step < 4 or step % 10 == 0 or step == 49
        model, losses, final_accuracy, mi_input, mi_output, mi_input_diffs, mi_output_diffs = train_on_dataset(
            dataset=(xs, ys),
            model=model,
            batch_size=32,
            learning_rate=1e-3,
            steps=1000,
            compute_mi_every=100 if compute_mi else 0,
        )
        final_losses.append(losses[-1])
        final_accuracies.append(final_accuracy)
        if compute_mi:
            mi_inputs.append(mi_input)
            mi_outputs.append(mi_output)
            mi_input_diffss.append(mi_input_diffs)
            mi_output_diffss.append(mi_output_diffs)

    return final_losses, final_accuracies, mi_inputs, mi_outputs, mi_input_diffss, mi_output_diffss


if __name__ == "__main__":
    final_losses_per_seed = []
    final_accuracies_per_seed = []
    mi_inputs_per_seed, mi_outputs_per_seed = [], []
    for seed in range(1):
        final_losses, final_accuracies, mi_inputs, mi_outputs, mi_input_diffs, mi_output_diffs = get_trajectory_losses(seed)
        final_losses_per_seed.append(final_losses)
        final_accuracies_per_seed.append(final_accuracies)
        mi_inputs_per_seed.append(mi_inputs)
        mi_outputs_per_seed.append(mi_outputs)

    mean = np.array(final_losses_per_seed).mean(0)
    ste = np.array(final_losses_per_seed).std(0) / np.sqrt(len(final_losses_per_seed))
    plt.grid(alpha=0.2)
    plt.plot(mean)
    plt.fill_between(range(len(mean)), mean - ste, mean + ste, alpha=0.3)
    plt.ylim(ymin=0)
    plt.xlabel('Trajectory Index')
    plt.ylabel('Final Loss')
    plt.title('Final Losses Across Different Input Bit Permutations')
    plt.savefig('final_losses.png')

    mean = np.array(final_accuracies_per_seed).mean(0)
    ste = np.array(final_accuracies_per_seed).std(0) / np.sqrt(len(final_accuracies_per_seed))
    plt.clf()
    plt.grid(alpha=0.2)
    plt.plot(mean)
    plt.fill_between(range(len(mean)), mean - ste, mean + ste, alpha=0.3)
    plt.ylim(0, 1.1)
    plt.xlabel('Trajectory Index')
    plt.ylabel('Final Accuracy')
    plt.title('Final Accuracies Across Different Input Bit Permutations')
    plt.savefig('final_accuracies.png')

    mi_inputs = np.array(mi_inputs_per_seed).mean(0)
    mi_outputs = np.array(mi_outputs_per_seed).mean(0)
    for traj in range(len(mi_inputs)):
        mi_input = mi_inputs[traj]
        mi_output = mi_outputs[traj]

        plot_mi_plane(mi_input, mi_output, idx=traj)

    plt.clf()
    plt.grid(alpha=0.2)
    for layer, diffs in mi_input_diffs[0].items():
        for neuron_idx, diff in enumerate(diffs):
            label = layer+f' Neuron {neuron_idx}'
            y = np.array([mi[layer][neuron_idx] for mi in mi_input_diffs])
            plt.plot(y, label=label)
    plt.xlabel('Iteration')
    plt.ylabel('MI Drop')
    plt.legend()
    plt.savefig('mi_input_drops.png')

    plt.clf()
    plt.grid(alpha=0.2)
    for layer, diffs in mi_output_diffs[0].items():
        for neuron_idx, diff in enumerate(diffs):
            label = layer+f' Neuron {neuron_idx}'
            y = np.array([mi[layer][neuron_idx] for mi in mi_output_diffs])
            plt.plot(y, label=label)
    plt.xlabel('Iteration')
    plt.ylabel('MI Drop')
    plt.legend()
    plt.savefig('mi_output_drops.png')
